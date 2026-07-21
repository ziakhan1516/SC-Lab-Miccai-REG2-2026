from typing import Optional

import torch
from torch.utils.data import DataLoader

from attention_mil import AttentionMIL
from multimodal_alignment import WSIReportGenerator
from wsi_dataset import WSIDataset, collate_fn


class WSIEmbeddingPipeline:
    """
    Generate slide-level embeddings from patch-level WSI features.

    Pipeline:
        Patch Features [N,1024]
                ↓
           Attention MIL
                ↓
        Slide Embedding [512]
    """

    def __init__(
        self,
        pt_dir,
        batch_size=1,
        num_workers=0,
        checkpoint_dir: Optional[str] = None,
        device=None,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.dataset = WSIDataset(pt_dir)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=str(self.device).startswith("cuda"),
        )

        if checkpoint_dir:
            generator = WSIReportGenerator.from_pretrained(
                checkpoint_dir,
                device=self.device,
            )
            self.model = generator.wsi_encoder
        else:
            print(
                "Warning: no checkpoint_dir was provided. "
                "Embeddings will come from a randomly initialized MIL encoder."
            )
            self.model = AttentionMIL(
                in_dim=1024,
                hidden_dim=512,
                attention_dim=128,
            ).to(self.device)

        self.model.eval()

    def generate_embeddings(self):

        all_embeddings = {}

        with torch.no_grad():

            for feats_list, slide_ids in self.loader:

                for feats, slide_id in zip(feats_list, slide_ids):

                    feats = feats.to(self.device)

                    # Forward pass
                    bag_embedding, attention_weights = self.model(feats)

                    # Save embedding
                    all_embeddings[slide_id] = {
                        "embedding": bag_embedding.cpu(),
                        "attention": attention_weights.cpu(),
                        "num_patches": feats.shape[0]
                    }

                    print(
                        f"{slide_id} | "
                        f"patches={feats.shape[0]} | "
                        f"embedding={bag_embedding.shape}"
                    )

        print(
            f"\nFinished processing "
            f"{len(all_embeddings)} slides"
        )

        return all_embeddings

    def save_embeddings(
        self,
        embeddings,
        save_path="wsi_slide_embeddings.pt"
    ):

        torch.save(embeddings, save_path)

        print(f"\nSaved embeddings → {save_path}")
