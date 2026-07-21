import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Transformers versions can probe a newer PyTorch pytree API even when the
# installed torch exposes the older private name. Keep the shim local and tiny.
if hasattr(torch.utils, "_pytree"):
    _pytree = torch.utils._pytree
    if (
        not hasattr(_pytree, "register_pytree_node")
        and hasattr(_pytree, "_register_pytree_node")
    ):
        def _register_pytree_node_compat(
            typ,
            flatten_fn,
            unflatten_fn,
            *,
            to_dumpable_context=None,
            from_dumpable_context=None,
            **_,
        ):
            return _pytree._register_pytree_node(
                typ,
                flatten_fn,
                unflatten_fn,
                to_dumpable_context=to_dumpable_context,
                from_dumpable_context=from_dumpable_context,
            )

        _pytree.register_pytree_node = _register_pytree_node_compat

from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_mil import AttentionMIL
from text_encoder import SFTTextEncoder


def load_generator(checkpoint_dir: str, device: Optional[str] = None):
    """Load a saved generator, dispatching on the `arch` field of its config
    ('causal' -> WSIReportGenerator, 'seq2seq' -> BioBART WSISeq2SeqReportGenerator)."""
    with (Path(checkpoint_dir) / "model_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("arch") == "reasoning":
        from reasoning_mllm import WSIReasoningReportGenerator

        return WSIReasoningReportGenerator.from_pretrained(checkpoint_dir, device=device)
    if config.get("arch") == "seq2seq":
        from seq2seq_alignment import WSISeq2SeqReportGenerator

        return WSISeq2SeqReportGenerator.from_pretrained(checkpoint_dir, device=device)
    return WSIReportGenerator.from_pretrained(checkpoint_dir, device=device)


class WSIReportGenerator(nn.Module):
    """
    WSI-conditioned causal language model for SFT.

    Patch features -> Attention MIL slide embedding -> learned soft prefix
    -> causal LM generates:
      Reasoning:
      Pathology Report:
    """

    def __init__(
        self,
        lm_name: str = "distilgpt2",
        feature_dim: int = 1024,
        mil_hidden_dim: int = 512,
        attention_dim: int = 128,
        prefix_length: int = 8,
        dropout: float = 0.1,
        max_prompt_length: int = 256,
        max_target_length: int = 1024,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        added_tokens = 0

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                added_tokens = self.tokenizer.add_special_tokens(
                    {"pad_token": "<|pad|>", "eos_token": "<|endoftext|>"}
                )

        self.language_model = AutoModelForCausalLM.from_pretrained(lm_name)
        if added_tokens:
            self.language_model.resize_token_embeddings(len(self.tokenizer))

        self.language_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_context_length = self._infer_context_length()
        max_text_length = self.max_context_length - prefix_length
        if max_text_length < 2:
            raise ValueError(
                f"prefix_length={prefix_length} leaves no room for text in a "
                f"context window of {self.max_context_length} tokens."
            )

        max_prompt_length = min(max_prompt_length, max_text_length - 1)
        max_target_length = min(
            max_target_length,
            max_text_length - 1,
        )

        self.config_dict = {
            "arch": "causal",
            "lm_name": lm_name,
            "feature_dim": feature_dim,
            "mil_hidden_dim": mil_hidden_dim,
            "attention_dim": attention_dim,
            "prefix_length": prefix_length,
            "dropout": dropout,
            "max_prompt_length": max_prompt_length,
            "max_target_length": max_target_length,
        }

        lm_hidden_dim = getattr(
            self.language_model.config,
            "hidden_size",
            getattr(self.language_model.config, "n_embd", None),
        )
        if lm_hidden_dim is None:
            raise ValueError("Could not infer language-model hidden size.")

        self.prefix_length = prefix_length
        self.wsi_encoder = AttentionMIL(
            in_dim=feature_dim,
            hidden_dim=mil_hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
        self.prefix_projection = nn.Sequential(
            nn.Linear(mil_hidden_dim, mil_hidden_dim),
            nn.Tanh(),
            nn.Linear(mil_hidden_dim, prefix_length * lm_hidden_dim),
        )
        self.text_encoder = SFTTextEncoder(
            tokenizer=self.tokenizer,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            max_text_length=max_text_length,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _infer_context_length(self) -> int:
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(self.language_model.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit

        return 1024

    def freeze_language_model(self) -> None:
        for param in self.language_model.parameters():
            param.requires_grad = False

    def enable_gradient_checkpointing(self) -> None:
        # Cuts activation memory for long sequences. use_reentrant=False is the
        # DDP-compatible variant; cache must be off during checkpointed training
        # (re-enabled explicitly at generation time).
        self.language_model.config.use_cache = False
        try:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model, "enable_input_require_grads"):
            self.language_model.enable_input_require_grads()

    def encode_wsi(self, features: List[torch.Tensor]) -> torch.Tensor:
        embeddings = []

        for feats in features:
            feats = feats.to(self.device)
            bag_embedding, _ = self.wsi_encoder(feats)
            embeddings.append(bag_embedding)

        return torch.stack(embeddings, dim=0)

    def _prefix_embeddings(self, features: List[torch.Tensor]) -> torch.Tensor:
        wsi_embeddings = self.encode_wsi(features)
        prefix = self.prefix_projection(wsi_embeddings)
        batch_size = prefix.shape[0]
        lm_hidden_dim = self.language_model.get_input_embeddings().embedding_dim
        return prefix.view(batch_size, self.prefix_length, lm_hidden_dim)

    def _build_inputs(
        self,
        features: List[torch.Tensor],
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        text_batch = self.text_encoder.encode(
            prompts=prompts,
            targets=targets,
            device=self.device,
        )

        token_embeddings = self.language_model.get_input_embeddings()(
            text_batch["input_ids"]
        )
        prefix_embeddings = self._prefix_embeddings(features)
        inputs_embeds = torch.cat([prefix_embeddings, token_embeddings], dim=1)

        prefix_mask = torch.ones(
            (len(prompts), self.prefix_length),
            dtype=text_batch["attention_mask"].dtype,
            device=self.device,
        )
        attention_mask = torch.cat(
            [prefix_mask, text_batch["attention_mask"]],
            dim=1,
        )

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "input_ids": text_batch["input_ids"],
        }

        if targets is not None:
            prefix_labels = torch.full(
                (len(prompts), self.prefix_length),
                -100,
                dtype=text_batch["labels"].dtype,
                device=self.device,
            )
            model_inputs["labels"] = torch.cat(
                [prefix_labels, text_batch["labels"]],
                dim=1,
            )

        return model_inputs

    def forward(
        self,
        features: List[torch.Tensor],
        prompts: List[str],
        targets: List[str],
    ):
        model_inputs = self._build_inputs(
            features=features,
            prompts=prompts,
            targets=targets,
        )

        return self.language_model(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
        )

    @torch.no_grad()
    def generate(
        self,
        features: List[torch.Tensor],
        prompts: List[str],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        self.eval()
        model_inputs = self._build_inputs(
            features=features,
            prompts=prompts,
            targets=None,
        )

        # inputs_embeds already includes the WSI soft prefix + prompt tokens.
        input_length = model_inputs["inputs_embeds"].shape[1]
        available_new_tokens = self.max_context_length - input_length
        if available_new_tokens < 1:
            raise ValueError(
                "The prompt plus WSI prefix already fills the language-model "
                "context window."
            )

        generate_kwargs = {
            "inputs_embeds": model_inputs["inputs_embeds"],
            "attention_mask": model_inputs["attention_mask"],
            "max_new_tokens": min(max_new_tokens, available_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # fast decoding even if checkpointing disabled it
        }
        if do_sample:
            generate_kwargs.update({"temperature": temperature, "top_p": top_p})

        # When `inputs_embeds` is supplied (and no `input_ids`), decoder-only
        # models return ONLY the newly generated token ids, so no slicing is
        # needed.
        sequences = self.language_model.generate(**generate_kwargs)

        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
        )

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
