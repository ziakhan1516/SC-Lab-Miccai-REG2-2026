import sys
sys.path.insert(0, "/home/ali/storage1/Bin-Version2/Reg2/codings/Try1")
import multimodal_alignment  # apply torch._pytree shim before transformers
import evaluate_metrics as em
sys.argv = [
    "evaluate_metrics.py",
    "--ground-truth", "ground_truth/metric_A/chain-of-thoughts-ground-truth.json",
    "--predictions", "data/predictions/interf1/predictions.json",
    "--visual-json", "data/predictions/interf0/predictions.json",
    "--visual-mapping-txt", "ground_truth/metric_B/rois_mapping.txt",
    "--judge-model-path", "ground_truth/Qwen3-8B",
    "--device", "cuda",
    "--output", "test/output/official_all_scores.json",
]
em.main()
