import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from multimodal_alignment import WSIReportGenerator
from multimodal_dataset import MultiModalWSIDataset, collate_fn


def _env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


class WSISFTTrainer:
    def __init__(
        self,
        processed_data: List[Dict[str, Any]],
        pt_dir: str,
        output_dir: str = "checkpoints/wsi_sft",
        model_name: str = "distilgpt2",
        arch: str = "causal",
        batch_size: int = 1,
        num_workers: int = 0,
        lr: float = 2e-5,
        epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        feature_dim: int = 1024,
        mil_hidden_dim: int = 512,
        attention_dim: int = 128,
        prefix_length: int = 8,
        num_visual_tokens: int = 32,
        resampler_depth: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        think_format: bool = True,
        load_in_4bit: bool = False,
        max_prompt_length: int = 256,
        max_target_length: int = 1024,
        freeze_language_model: bool = False,
        missing_wsi: str = "error",
        device: Optional[str] = None,
        device_map=None,
        max_memory=None,
        ddp_find_unused_parameters: bool = False,
        static_graph: bool = False,
        gradient_checkpointing: bool = False,
        eval_samples: Optional[List[Dict[str, Any]]] = None,
        eval_max_new_tokens: int = 1024,
        eval_use_embeddings: bool = False,
        eval_embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ):
        self.output_dir = output_dir
        self.epochs = epochs
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.eval_samples = eval_samples or []
        self.eval_max_new_tokens = eval_max_new_tokens
        self.eval_use_embeddings = eval_use_embeddings
        self.eval_embedding_model = eval_embedding_model
        self._scorer = None
        self.distributed = _env_world_size() > 1
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        if self.distributed:
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = f"cuda:{self.local_rank}"
            else:
                self.device = "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.is_main_process = self.rank == 0

        self._log(
            f"Using device: {self.device} "
            f"| distributed={self.distributed} "
            f"| world_size={self.world_size}"
        )

        self.dataset = MultiModalWSIDataset(
            json_data=processed_data,
            pt_dir=pt_dir,
            missing=missing_wsi,
        )

        self.sampler = None
        if self.distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=str(self.device).startswith("cuda"),
        )

        if arch == "reasoning":
            from reasoning_mllm import WSIReasoningReportGenerator

            base_model = WSIReasoningReportGenerator(
                lm_name=model_name,
                feature_dim=feature_dim,
                num_visual_tokens=num_visual_tokens,
                resampler_depth=resampler_depth,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                think_format=think_format,
                load_in_4bit=load_in_4bit,
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
                device_map=device_map,
                max_memory=max_memory,
            )
            if base_model.dispatched:
                # Quantized (4-bit) or model-parallel: the LLM is already placed
                # on its GPU(s); moving it would break bitsandbytes/accelerate.
                # Tensors are built on the embedding device instead.
                self.device = str(base_model.input_device)
                self._log(f"Base LLM pre-placed (4bit={load_in_4bit}, "
                          f"device_map={device_map}); input device {self.device}")
            else:
                base_model = base_model.to(self.device)
        elif arch == "seq2seq":
            from seq2seq_alignment import WSISeq2SeqReportGenerator

            base_model = WSISeq2SeqReportGenerator(
                lm_name=model_name,
                feature_dim=feature_dim,
                num_visual_tokens=num_visual_tokens,
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
            ).to(self.device)
        else:
            base_model = WSIReportGenerator(
                lm_name=model_name,
                feature_dim=feature_dim,
                mil_hidden_dim=mil_hidden_dim,
                attention_dim=attention_dim,
                prefix_length=prefix_length,
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
            ).to(self.device)

        if freeze_language_model:
            base_model.freeze_language_model()

        if gradient_checkpointing:
            base_model.enable_gradient_checkpointing()
            self._log("Gradient checkpointing enabled.")

        if self.distributed:
            ddp_kwargs = {}
            if str(self.device).startswith("cuda"):
                ddp_kwargs = dict(
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            # static_graph is the correct DDP mode for gradient checkpointing:
            # checkpointing reruns forward in backward, which makes the reducer
            # mark params "ready twice" under find_unused_parameters. static_graph
            # traces the (fixed LoRA+resampler) graph once and handles this. The
            # two options are mutually exclusive.
            if static_graph:
                self.model = DistributedDataParallel(
                    base_model, static_graph=True, **ddp_kwargs
                )
            else:
                self.model = DistributedDataParallel(
                    base_model,
                    find_unused_parameters=ddp_find_unused_parameters,
                    **ddp_kwargs,
                )
        else:
            self.model = base_model

        trainable_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        if not trainable_params:
            raise ValueError("No trainable parameters are available.")

        self.optimizer = optim.AdamW(trainable_params, lr=lr)

        self.training_args = {
            "pt_dir": pt_dir,
            "output_dir": output_dir,
            "model_name": model_name,
            "arch": arch,
            "num_visual_tokens": num_visual_tokens,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "lr": lr,
            "epochs": epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "feature_dim": feature_dim,
            "mil_hidden_dim": mil_hidden_dim,
            "attention_dim": attention_dim,
            "prefix_length": prefix_length,
            "max_prompt_length": max_prompt_length,
            "max_target_length": max_target_length,
            "freeze_language_model": freeze_language_model,
            "missing_wsi": missing_wsi,
            "num_samples": len(self.dataset),
            "distributed": self.distributed,
            "world_size": self.world_size,
            "ddp_find_unused_parameters": ddp_find_unused_parameters,
        }

    def train(self) -> None:
        global_step = 0
        self.model.train()

        for epoch in range(self.epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            total_loss = 0.0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.loader):
                outputs = self.model(
                    features=batch["features"],
                    prompts=batch["prompt"],
                    targets=batch["target_text"],
                )

                raw_loss = outputs.loss
                loss = raw_loss / self.gradient_accumulation_steps
                loss.backward()

                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(self.loader)
                )

                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                total_loss += raw_loss.item()

                self._log(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.loader)}] "
                    f"Loss: {raw_loss.item():.4f}"
                )

            avg_loss = self._epoch_average_loss(total_loss, len(self.loader))
            self._log(
                f"\nEpoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}\n"
            )
            self.save_model(Path(self.output_dir) / f"epoch_{epoch + 1}")
            self.evaluate_after_epoch(epoch + 1)

        self.save_model(self.output_dir)
        self._log("\nSFT training completed.")

    def evaluate_after_epoch(self, epoch_number: int) -> None:
        """Generate on a few held-out test cases and score them with the
        Workflow Reasoning metric. Runs on rank 0 only; other ranks wait."""
        if not self.eval_samples:
            return
        if not self.is_main_process:
            self._barrier()
            return

        # Imported lazily so non-eval training never pays the import cost.
        from evaluate_workflow_reasoning import (
            WorkflowReasoningMetrics,
            TextEmbedder,
            ground_truth_from_sample,
        )
        from wsi_dataset import load_wsi_features, normalize_slide_id

        if self._scorer is None:
            embedder = TextEmbedder(
                model_name=self.eval_embedding_model,
                device=self.device,
                disabled=not self.eval_use_embeddings,
            )
            self._scorer = WorkflowReasoningMetrics(embedder)

        model = self._model_to_save()
        was_training = model.training
        model.eval()

        file_map = self.dataset.file_map
        results = []
        print(
            f"\n{'#'*80}\n# Workflow Reasoning eval after epoch {epoch_number} "
            f"on {len(self.eval_samples)} random test cases\n{'#'*80}"
        )

        for idx, sample in enumerate(self.eval_samples, start=1):
            slide_id = normalize_slide_id(sample["slide_id"])
            file_path = file_map.get(slide_id)
            if file_path is None:
                continue

            features = load_wsi_features(str(file_path))
            generated = model.generate(
                features=[features],
                prompts=[sample["prompt"]],
                max_new_tokens=self.eval_max_new_tokens,
                do_sample=False,
            )[0].strip()

            gt_steps, gt_report = ground_truth_from_sample(sample)
            case_metrics = self._scorer.score_case(
                generated,
                sample["target_text"],
                gt_steps=gt_steps,
                gt_report=gt_report,
            )
            results.append(
                {
                    "id": sample["id"],
                    "slide_id": slide_id,
                    "generated_text": generated,
                    "reference_text": sample["target_text"],
                    "metrics": case_metrics,
                }
            )

            print(f"\n{'='*80}\n[{idx}/{len(self.eval_samples)}] {slide_id}")
            print(f"{'-'*80}\n--- GENERATED ---\n{generated}")
            print(f"\n--- REFERENCE ---\n{sample['target_text']}")
            print(
                f"\n--- SCORES ---  "
                f"BPV={case_metrics['binary_path_validity']:.3f} "
                f"EdgeF1={case_metrics['edge_f1']['f1']:.3f} "
                f"MESS={case_metrics['mess']:.3f} "
                f"FinalReport={case_metrics['final_report_score']['score']:.3f} "
                f"WorkflowReasoning={case_metrics['workflow_reasoning_score']:.3f}"
            )

        if results:
            def _avg(key_fn):
                vals = [key_fn(r["metrics"]) for r in results]
                return sum(vals) / len(vals)

            summary = {
                "binary_path_validity": _avg(lambda m: m["binary_path_validity"]),
                "edge_f1": _avg(lambda m: m["edge_f1"]["f1"]),
                "mess": _avg(lambda m: m["mess"]),
                "final_report_score": _avg(lambda m: m["final_report_score"]["score"]),
                "workflow_reasoning_score": _avg(
                    lambda m: m["workflow_reasoning_score"]
                ),
            }
            print(f"\n{'#'*80}\n# Epoch {epoch_number} eval summary:")
            print(json.dumps(summary, indent=2))

            payload = {
                "epoch": epoch_number,
                "num_samples": len(results),
                "embeddings": self.eval_embedding_model
                if self.eval_use_embeddings
                else "lexical-fallback",
                "summary": summary,
                "results": results,
            }
            out_path = Path(self.output_dir) / f"eval_epoch_{epoch_number}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"# Saved in-training eval results: {out_path}\n{'#'*80}\n")

        if was_training:
            model.train()
        self._barrier()

    def save_model(self, save_dir: str) -> None:
        if not self.is_main_process:
            self._barrier()
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_to_save = self._model_to_save()
        model_to_save.save_pretrained(str(save_path))

        with (save_path / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(self.training_args, f, indent=2)

        print(f"Saved SFT model: {save_path}")
        self._barrier()

    def _model_to_save(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def _log(self, message: str) -> None:
        if self.is_main_process:
            print(message)

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _epoch_average_loss(self, total_loss: float, num_batches: int) -> float:
        if not self.distributed:
            return total_loss / max(1, num_batches)

        loss_tensor = torch.tensor(
            [total_loss, float(num_batches)],
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor[0] / loss_tensor[1].clamp_min(1.0)).item()


# Backward-compatible name for old scripts that imported AlignmentTrainer.
AlignmentTrainer = WSISFTTrainer
