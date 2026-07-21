"""BioBART (encoder-decoder) WSI report generator.

Why encoder-decoder instead of the causal soft-prefix model:
  - The decoder cross-attends to the WSI visual tokens at *every* step, which
    grounds the answers/report in the slide far better than a fixed prefix.
  - BioBART is pretrained on biomedical text, improving report phrasing
    (keyword overlap + embedding cosine in the Final Report Score).

Pipeline:
  patch features -> Perceiver-style attention pool -> K visual tokens
  -> [visual tokens ; instruction-prompt embeddings] feed the BART encoder
  -> BART decoder generates "Reasoning:" + "Pathology Report:".

BART caps the decoder at 1024 positions. For the ~11% of cases whose target
exceeds that, `_budget_label_ids` keeps the full final report and truncates the
*reasoning* instead, so the 40%-weighted report is always learned.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Importing this first applies the torch._pytree compatibility shim before any
# transformers import (no-op on newer torch).
import multimodal_alignment  # noqa: F401
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPORT_MARKER = "Pathology Report:"


class PerceiverPool(nn.Module):
    """Pool a variable number of patch features into K fixed visual tokens via
    cross-attention from learned query vectors."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.queries = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.num_tokens = num_tokens
        self.d_model = d_model

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for feats in features:
            x = self.proj(feats).unsqueeze(0)            # [1, N, d]
            q = self.queries.unsqueeze(0)                # [1, K, d]
            pooled, _ = self.attn(q, x, x)               # [1, K, d]
            outs.append(self.norm(pooled.squeeze(0)))    # [K, d]
        return torch.stack(outs, dim=0)                  # [B, K, d]


class WSISeq2SeqReportGenerator(nn.Module):
    ARCH = "seq2seq"

    def __init__(
        self,
        lm_name: str = "GanjinZero/biobart-v2-base",
        feature_dim: int = 1024,
        num_visual_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_prompt_length: int = 200,
        max_target_length: int = 1022,
        **_ignored,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(lm_name)
        config = self.model.config

        self.d_model = config.d_model
        self.embed_scale = (
            math.sqrt(self.d_model) if getattr(config, "scale_embedding", False) else 1.0
        )

        max_pos = getattr(config, "max_position_embeddings", 1024)
        self.num_visual_tokens = num_visual_tokens
        self.max_target_length = min(max_target_length, max_pos - 2)
        self.max_prompt_length = min(
            max_prompt_length, max_pos - num_visual_tokens - 2
        )

        self.visual = PerceiverPool(
            in_dim=feature_dim,
            d_model=self.d_model,
            num_tokens=num_visual_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.config_dict = {
            "arch": self.ARCH,
            "lm_name": lm_name,
            "feature_dim": feature_dim,
            "num_visual_tokens": num_visual_tokens,
            "num_heads": num_heads,
            "dropout": dropout,
            "max_prompt_length": self.max_prompt_length,
            "max_target_length": self.max_target_length,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze_language_model(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def enable_gradient_checkpointing(self) -> None:
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.model.gradient_checkpointing_enable()

    def _encoder_inputs(self, features: List[torch.Tensor], prompts: List[str]):
        visual_tokens = self.visual([f.to(self.device) for f in features])  # [B,K,d]

        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_mask = encoded["attention_mask"].to(self.device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids) * self.embed_scale

        inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)
        visual_mask = torch.ones(
            visual_tokens.shape[0],
            visual_tokens.shape[1],
            dtype=prompt_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([visual_mask, prompt_mask], dim=1)
        return inputs_embeds, attention_mask

    def _budget_label_ids(self, target: str) -> List[int]:
        """Tokenize a target, keeping the final report whole when truncation is
        required (truncates reasoning instead)."""
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        limit = self.max_target_length

        full = self.tokenizer(target, add_special_tokens=True).input_ids
        if len(full) <= limit:
            return full

        idx = target.rfind(REPORT_MARKER)
        if idx == -1:
            return self.tokenizer(
                target, add_special_tokens=True, truncation=True, max_length=limit
            ).input_ids

        head_text = target[:idx]
        report_text = target[idx:]
        report_ids = self.tokenizer(report_text, add_special_tokens=False).input_ids
        report_ids = report_ids[: max(0, limit - 2)]
        head_budget = max(0, limit - 2 - len(report_ids))
        head_ids = self.tokenizer(head_text, add_special_tokens=False).input_ids[:head_budget]

        ids = [t for t in (bos,) if t is not None] + head_ids + report_ids
        if eos is not None:
            ids = ids + [eos]
        return ids[:limit]

    def _labels(self, targets: List[str]) -> torch.Tensor:
        rows = [self._budget_label_ids(t) for t in targets]
        max_len = max(len(r) for r in rows)
        # -100 padding is ignored by the loss; BART's shift_tokens_right maps it
        # to pad for the decoder inputs.
        padded = [r + [-100] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def forward(self, features, prompts, targets):
        inputs_embeds, attention_mask = self._encoder_inputs(features, prompts)
        labels = self._labels(targets)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        features,
        prompts,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
    ) -> List[str]:
        self.eval()
        inputs_embeds, attention_mask = self._encoder_inputs(features, prompts)

        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": min(max_new_tokens, self.max_target_length),
            "do_sample": do_sample,
            "num_beams": num_beams,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": num_beams > 1,
        }
        if do_sample:
            generate_kwargs.update({"temperature": temperature, "top_p": top_p})

        sequences = self.model.generate(**generate_kwargs)
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

    def save_pretrained(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with (output_path / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config_dict, f, indent=2)

        torch.save(
            {"model_state_dict": self.state_dict(), "config": self.config_dict},
            output_path / "model.pt",
        )
        self.tokenizer.save_pretrained(output_path / "tokenizer")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: Optional[str] = None):
        checkpoint_path = Path(checkpoint_dir)
        with (checkpoint_path / "model_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("arch", None)
        model = cls(**config)
        checkpoint = torch.load(
            checkpoint_path / "model.pt",
            map_location=device or "cpu",
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if device:
            model = model.to(device)

        return model
