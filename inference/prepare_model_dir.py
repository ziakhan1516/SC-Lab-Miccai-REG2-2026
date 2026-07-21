"""Assemble the model/ directory that becomes /opt/ml/model in the container.

Because the container has NO internet, everything the loader needs must live
here (this is what do_save.sh packs into model.tar.gz):

    model/
    ├── base_llm/                 full DeepSeek-R1-Distill-Qwen-1.5B (offline)
    ├── generation/               trained checkpoint
    │   ├── model_config.json
    │   ├── lora_adapter/
    │   ├── resampler.pt
    │   └── tokenizer/
    └── conch/
        └── pytorch_model.bin     CONCH ViT-B-16 weights

Run from the submission template dir (any Python with `transformers` works for
exporting the base LLM, e.g. the training env):

    python prepare_model_dir.py \
        --checkpoint /home/ali/storage1/Bin-Version2/Reg2/codings/Try2/checkpoints/wsi_reasoning_r1qwen1p5b_conch \
        --conch      /home/ali/storage1/Bin-Version2/Reg2/codings/Try2/ClamforMiccai2026/pytorch_model.bin
"""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = "/home/ali/storage1/Bin-Version2/Reg2/codings/Try2/checkpoints/wsi_reasoning_r1qwen1p5b_conch"
DEFAULT_CONCH = "/home/ali/storage1/Bin-Version2/Reg2/codings/Try2/ClamforMiccai2026/pytorch_model.bin"


def _copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied {src} -> {dst}")


def export_base_llm(lm_name: str, dst: Path):
    """Save the base LLM + tokenizer as a plain offline directory (bf16, the
    dtype it is loaded in at runtime — halves the tarball vs fp32)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  exporting base LLM '{lm_name}' (bf16) -> {dst} (uses local HF cache)")
    tok = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.bfloat16)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))
    tok.save_pretrained(str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Trained generation checkpoint.")
    ap.add_argument("--conch", default=DEFAULT_CONCH, help="CONCH pytorch_model.bin.")
    ap.add_argument("--model-dir", default=str(HERE / "model"), help="Output model/ dir.")
    ap.add_argument("--skip-base-llm", action="store_true",
                    help="Do not re-export base_llm/ (if already present).")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    model_dir = Path(args.model_dir)
    gen_dir = model_dir / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print("Assembling model/ ...")

    # 1) Generation checkpoint (config + adapter + resampler + tokenizer).
    with (ckpt / "model_config.json").open() as f:
        config = json.load(f)
    lm_name = config["lm_name"]
    # Point lm_name at the bundled offline copy so from-scratch loads need no net.
    config["lm_name"] = "/opt/ml/model/base_llm"
    with (gen_dir / "model_config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"  wrote {gen_dir/'model_config.json'} (lm_name -> /opt/ml/model/base_llm)")

    _copytree(ckpt / "lora_adapter", gen_dir / "lora_adapter")
    _copytree(ckpt / "tokenizer", gen_dir / "tokenizer")
    shutil.copy2(ckpt / "resampler.pt", gen_dir / "resampler.pt")
    print(f"  copied resampler.pt")

    # 2) Base LLM (full weights, offline).
    if not args.skip_base_llm:
        export_base_llm(lm_name, model_dir / "base_llm")
    else:
        print("  skipped base_llm export")

    # 3) CONCH encoder weights.
    conch_dst = model_dir / "conch"
    conch_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.conch, conch_dst / "pytorch_model.bin")
    print(f"  copied CONCH -> {conch_dst/'pytorch_model.bin'}")

    print("\nDone. model/ is ready. Pack it with ./do_save.sh (-> model.tar.gz).")


if __name__ == "__main__":
    main()
