"""WSI-conditioned reasoning LLM (true generation, LLaVA/BLIP-2 style).

Why this replaces the distilgpt2 / BioBART generators
-----------------------------------------------------
The earlier generators were weak *language* models (distilgpt2: 1k ctx, no
reasoning; BioBART: 170M, 1k ctx), so the 40%-weighted final report and the
long 30-step chain-of-thought were never learned well. Here the decoder is a
real **reasoning LLM** (DeepSeek-R1-Distill-Qwen, native <think> reasoning)
that *generates* the whole answer, grounded on the slide:

    patch features [N,1024]
        -> PerceiverResampler -> K visual tokens in the LLM embedding space
        -> [visual tokens ; chat(system, user-instruction)]  (inputs_embeds)
        -> DeepSeek-R1-Distill-Qwen (LoRA) autoregressively writes:
               Reasoning:
               1. Question: ... / Answer: ... / Next Question: ...
               ...
               Pathology Report:
               <structured report>

The LLM is adapted with LoRA (parameter-efficient); the resampler is trained
fully. Supervision is applied only to the assistant tokens. Output text is in
the exact format the Workflow-Reasoning metric parser consumes, so nothing
downstream changes.

Interface parity with the other generators (so train.py / main.py / the
in-training metric eval all work unchanged):
    forward(features, prompts, targets) -> output with .loss
    generate(features, prompts, max_new_tokens, do_sample, temperature, top_p)
    save_pretrained / from_pretrained / freeze_language_model
    enable_gradient_checkpointing
"""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# A torch._pytree compatibility shim is only needed for transformers 4.46 on
# torch 2.1. The container ships newer torch/transformers where it is a no-op,
# so the original side-effect import is optional here.
try:  # pragma: no cover
    import multimodal_alignment  # noqa: F401
except Exception:
    pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

REPORT_MARKER = "Pathology Report:"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert pathologist. You are shown one whole-slide image (WSI) "
    "represented by learned visual feature tokens. First, think step by step "
    "inside <think> </think>: lay out the explicit, pathologist-style diagnostic "
    "workflow as numbered Question / Answer / Next Question steps, using the "
    "canonical workflow question wording. After </think>, write the final "
    "structured pathology report under a 'Pathology Report:' heading. Base every "
    "conclusion strictly on the visual evidence in the slide."
)

# The reasoning chain lives inside <think>; the final report follows </think>.
# This matches the DeepSeek-R1-Distill native format while staying parseable by
# the Workflow-Reasoning metric (it reads Question/Answer/Next lines anywhere and
# the report after the 'Pathology Report:' marker).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_REASONING_HEADER_RE = re.compile(r"^\s*Reasoning\s*:\s*", re.IGNORECASE)
_TRAILING_THINK_RE = re.compile(r"<think>\s*$")

# LoRA on every attention + MLP projection — the standard Qwen2 target set.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class PerceiverResampler(nn.Module):
    """Resample a variable-length bag of patch features into K fixed visual
    tokens with stacked cross-attention from learned latent queries (Flamingo /
    BLIP-2 style). Cross-attention at every layer grounds the tokens in the
    actual patches rather than a single pooled vector."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_tokens: int = 64,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.latents = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)

        self.cross_layers = nn.ModuleList()
        self.self_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.self_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(depth):
            self.cross_layers.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            self.self_layers.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            self.cross_norms.append(nn.LayerNorm(d_model))
            self.self_norms.append(nn.LayerNorm(d_model))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(d_model))

        self.out_norm = nn.LayerNorm(d_model)
        self.num_tokens = num_tokens
        self.d_model = d_model

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """features: list of [N_i, in_dim] tensors -> [B, K, d_model]."""
        outs = []
        for feats in features:
            x = self.input_proj(feats).unsqueeze(0)          # [1, N, d]
            q = self.latents.unsqueeze(0)                    # [1, K, d]
            for ci, si, cn, sn, ffn, fn in zip(
                self.cross_layers, self.self_layers, self.cross_norms,
                self.self_norms, self.ffns, self.ffn_norms,
            ):
                attended, _ = ci(cn(q), x, x)                # cross-attend to patches
                q = q + attended
                sa, _ = si(sn(q), sn(q), sn(q))              # latent self-attention
                q = q + sa
                q = q + ffn(fn(q))
            outs.append(self.out_norm(q.squeeze(0)))         # [K, d]
        return torch.stack(outs, dim=0)                      # [B, K, d]


class WSIReasoningReportGenerator(nn.Module):
    ARCH = "reasoning"

    def __init__(
        self,
        lm_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        feature_dim: int = 1024,
        num_visual_tokens: int = 576,
        resampler_depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_prompt_length: int = 1024,
        max_target_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        think_format: bool = True,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        device_map=None,
        max_memory=None,
        _build_peft: bool = True,
        **_ignored,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.compute_dtype = dtype
        # device_map (e.g. "auto") shards the LLM layers across the visible GPUs
        # via accelerate hooks — needed for 8B/14B models that don't fit on one
        # 24 GB card. The resampler then lives on the embedding's device and
        # accelerate moves hidden states between GPUs during forward.
        self.load_in_4bit = load_in_4bit
        self.dispatched = device_map is not None
        load_kwargs = dict(torch_dtype=dtype)

        # QLoRA: load the frozen base LLM in 4-bit (NF4). This shrinks a 7B from
        # ~14 GB to ~4-5 GB so it fits a single 24 GB GPU; the LoRA adapters and
        # resampler still train in bf16 on top. 4-bit weights must be created
        # directly on the GPU, so we pin the model to the current CUDA device.
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            if device_map is None:
                device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
                device_map = {"": device_index}
            self.dispatched = True  # never .to() a quantized model; it's already placed

        if self.dispatched:
            load_kwargs["device_map"] = device_map
            if max_memory:
                load_kwargs["max_memory"] = max_memory
        base_llm = AutoModelForCausalLM.from_pretrained(lm_name, **load_kwargs)
        base_llm.config.pad_token_id = self.tokenizer.pad_token_id
        base_llm.config.use_cache = False

        if _build_peft:
            if load_in_4bit:
                # Casts layernorms to fp32 and enables input grads for stable
                # k-bit training (we drive gradient checkpointing ourselves via
                # enable_gradient_checkpointing(), so don't enable it here).
                from peft import prepare_model_for_kbit_training

                base_llm = prepare_model_for_kbit_training(
                    base_llm, use_gradient_checkpointing=False
                )
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(base_llm, lora_cfg)
        else:
            self.llm = base_llm  # adapter loaded later in from_pretrained

        d_model = base_llm.config.hidden_size
        self.d_model = d_model
        self.system_prompt = system_prompt
        self.think_format = think_format
        self.num_visual_tokens = num_visual_tokens
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length

        # Resampler kept in fp32 for stable optimisation; its output is cast to
        # the LLM embedding dtype right before concatenation.
        self.resampler = PerceiverResampler(
            in_dim=feature_dim,
            d_model=d_model,
            num_tokens=num_visual_tokens,
            depth=resampler_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # When the LLM is sharded, pin the resampler (and all tensors we build)
        # to the device that holds the input-embedding matrix.
        # Run the resampler in the LLM compute dtype (bf16): the cross-attention
        # over thousands of patches with hundreds of latent queries is the main
        # activation spike on the embedding GPU, so fp32 here doubles that cost.
        self.resampler.to(dtype=self.compute_dtype)
        self.input_device = self.llm.get_input_embeddings().weight.device
        if self.dispatched:
            self.resampler.to(self.input_device)

        self.config_dict = {
            "arch": self.ARCH,
            "lm_name": lm_name,
            "feature_dim": feature_dim,
            "num_visual_tokens": num_visual_tokens,
            "resampler_depth": resampler_depth,
            "num_heads": num_heads,
            "dropout": dropout,
            "max_prompt_length": max_prompt_length,
            "max_target_length": max_target_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "system_prompt": system_prompt,
            "think_format": think_format,
            "torch_dtype": torch_dtype,
            "load_in_4bit": load_in_4bit,
        }

    # ------------------------------------------------------------------ utils
    @property
    def device(self):
        # All input tensors (visual tokens, prompt/target embeds) must sit on the
        # embedding device; accelerate routes the rest across GPUs.
        if getattr(self, "dispatched", False):
            return self.input_device
        return next(self.llm.parameters()).device

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def _visual_tokens(self, features: List[torch.Tensor]) -> torch.Tensor:
        feats = [f.to(self.device, dtype=self.compute_dtype) for f in features]
        vis = self.resampler(feats)                       # [B, K, d] in compute dtype
        return vis.to(self.compute_dtype)

    def _prompt_text(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # R1-Distill templates force an opening "<think>" after the assistant
        # header. We supply <think> ourselves in the target, so strip the forced
        # one to avoid a duplicate (no-op for templates that don't add it).
        if self.think_format:
            text = _TRAILING_THINK_RE.sub("", text)
        return text

    def _format_target(self, target_text: str) -> str:
        """Reformat 'Reasoning:\\n... \\n\\nPathology Report:\\n...' into the
        <think> reasoning </think> + report layout (R1 native)."""
        if not self.think_format:
            return target_text
        idx = target_text.find(REPORT_MARKER)
        if idx == -1:
            reasoning, report = target_text, ""
        else:
            reasoning = target_text[:idx]
            report = target_text[idx + len(REPORT_MARKER):]
        reasoning = _REASONING_HEADER_RE.sub("", reasoning).strip()
        report = report.strip()
        return (
            f"{_THINK_OPEN}\n{reasoning}\n{_THINK_CLOSE}\n\n"
            f"{REPORT_MARKER}\n{report}"
        )

    def _prompt_ids(self, user_prompt: str) -> List[int]:
        text = self._prompt_text(user_prompt)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > self.max_prompt_length:
            # Keep the tail (task spec + output format live at the end).
            ids = ids[-self.max_prompt_length:]
        return ids

    def _target_ids(self, target_text: str) -> List[int]:
        """Tokenize the assistant target, ending with the chat-turn EOS. If it
        exceeds the budget, keep the whole final report and truncate the
        reasoning instead (the report is 40% of the score)."""
        eos = self.tokenizer.eos_token_id
        limit = self.max_target_length
        target_text = self._format_target(target_text)
        body = self.tokenizer(target_text, add_special_tokens=False).input_ids
        if len(body) <= limit - 1:
            return body + [eos]

        idx = target_text.rfind(REPORT_MARKER)
        if idx == -1:
            return body[: limit - 1] + [eos]
        head = self.tokenizer(target_text[:idx], add_special_tokens=False).input_ids
        report = self.tokenizer(target_text[idx:], add_special_tokens=False).input_ids
        report = report[: limit - 1]
        head = head[: max(0, limit - 1 - len(report))]
        return head + report + [eos]

    # ----------------------------------------------------------------- forward
    def _assemble_training_batch(self, features, prompts, targets):
        visual = self._visual_tokens(features)            # [B, K, d]
        k = visual.shape[1]

        seqs, masks, labels = [], [], []
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            p_ids = self._prompt_ids(prompt)
            t_ids = self._target_ids(target)
            ids = torch.tensor(p_ids + t_ids, device=self.device, dtype=torch.long)
            text_emb = self._embed_tokens(ids)            # [L, d]
            seq = torch.cat([visual[i], text_emb], dim=0)  # [K+L, d]
            label = torch.tensor(
                [-100] * (k + len(p_ids)) + t_ids,
                device=self.device, dtype=torch.long,
            )
            mask = torch.ones(seq.shape[0], device=self.device, dtype=torch.long)
            seqs.append(seq)
            labels.append(label)
            masks.append(mask)

        max_len = max(s.shape[0] for s in seqs)
        pad_emb = self._embed_tokens(
            torch.tensor([self.tokenizer.pad_token_id], device=self.device)
        )[0]
        inputs_embeds, attention_mask, label_batch = [], [], []
        for seq, mask, label in zip(seqs, masks, labels):
            pad = max_len - seq.shape[0]
            if pad:
                seq = torch.cat([seq, pad_emb.unsqueeze(0).expand(pad, -1)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad, device=self.device, dtype=torch.long)])
                label = torch.cat([label, torch.full((pad,), -100, device=self.device, dtype=torch.long)])
            inputs_embeds.append(seq)
            attention_mask.append(mask)
            label_batch.append(label)

        return (
            torch.stack(inputs_embeds, 0),
            torch.stack(attention_mask, 0),
            torch.stack(label_batch, 0),
        )

    def forward(self, features, prompts, targets):
        inputs_embeds, attention_mask, labels = self._assemble_training_batch(
            features, prompts, targets
        )
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return SimpleNamespace(loss=out.loss, logits=out.logits)

    # ---------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        features,
        prompts,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
    ) -> List[str]:
        was_training = self.training
        self.eval()
        visual = self._visual_tokens(features)            # [B, K, d]
        k = visual.shape[1]

        seqs, masks = [], []
        for i, prompt in enumerate(prompts):
            p_ids = torch.tensor(self._prompt_ids(prompt), device=self.device, dtype=torch.long)
            text_emb = self._embed_tokens(p_ids)
            seqs.append(torch.cat([visual[i], text_emb], dim=0))
            masks.append(torch.ones(k + p_ids.shape[0], device=self.device, dtype=torch.long))

        # Left-pad for batched decoding so every sequence ends at the same column.
        max_len = max(s.shape[0] for s in seqs)
        pad_emb = self._embed_tokens(
            torch.tensor([self.tokenizer.pad_token_id], device=self.device)
        )[0]
        embeds, attn = [], []
        for seq, mask in zip(seqs, masks):
            pad = max_len - seq.shape[0]
            if pad:
                seq = torch.cat([pad_emb.unsqueeze(0).expand(pad, -1), seq], dim=0)
                mask = torch.cat([torch.zeros(pad, device=self.device, dtype=torch.long), mask])
            embeds.append(seq)
            attn.append(mask)
        inputs_embeds = torch.stack(embeds, 0)
        attention_mask = torch.stack(attn, 0)

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        # Wall-clock safety net: if decoding runs much longer than a healthy
        # full-length report should take, stop gracefully and return whatever was
        # produced so far (still parseable) instead of letting one runaway case
        # blow the platform's per-case limit and halt the whole submission. The
        # budget is generous (default 180s, env REG_GEN_TIME_BUDGET_S; 0 disables)
        # so normal full-length generations are never truncated.
        budget_s = float(os.environ.get("REG_GEN_TIME_BUDGET_S", 180))
        if budget_s > 0:
            try:
                from transformers import StoppingCriteriaList, MaxTimeCriteria
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [MaxTimeCriteria(max_time=budget_s)]
                )
            except Exception:
                pass  # older transformers without MaxTimeCriteria -> no-op

        # With inputs_embeds (and no input_ids) a decoder-only model returns
        # ONLY the newly generated tokens (transformers 4.46) — no slicing.
        sequences = self.llm.generate(**gen_kwargs)
        text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        if was_training:
            self.train()
        return [t.strip() for t in text]

    # ----------------------------------------------------------- train helpers
    def freeze_language_model(self) -> None:
        # The LLM is already frozen by LoRA (only adapters train). Provided for
        # interface parity; calling it additionally freezes nothing extra.
        for name, param in self.llm.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def enable_gradient_checkpointing(self) -> None:
        self.llm.config.use_cache = False
        base = self.llm
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    # -------------------------------------------------------------------- I/O
    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config_dict, f, indent=2)
        # LoRA adapter only (small) + the trained resampler.
        self.llm.save_pretrained(str(out / "lora_adapter"))
        torch.save(self.resampler.state_dict(), out / "resampler.pt")
        self.tokenizer.save_pretrained(str(out / "tokenizer"))

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: Optional[str] = None):
        ckpt = Path(checkpoint_dir)
        with (ckpt / "model_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        config.pop("arch", None)

        model = cls(_build_peft=False, **config)
        # Reload the base LLM wrapped with the saved LoRA adapter.
        model.llm = PeftModel.from_pretrained(model.llm, str(ckpt / "lora_adapter"))
        model.resampler.load_state_dict(
            torch.load(ckpt / "resampler.pt", map_location="cpu")
        )
        if (ckpt / "tokenizer").exists():
            model.tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"), use_fast=True)
        if device:
            model = model.to(device)
        return model
