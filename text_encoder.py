from typing import Dict, List, Optional

import torch


class SFTTextEncoder:
    """
    Builds causal-LM inputs for supervised fine-tuning.

    Labels are masked for the prompt tokens, so loss is computed only on the
    desired pathology reasoning/report answer.
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 256,
        max_target_length: int = 1024,
        max_text_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.max_text_length = max_text_length

    def _encode_text(self, text: str, max_length: int) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

    def _encode_prompt(self, prompt: str, max_length: int) -> List[int]:
        return self._encode_text(prompt, max_length=max_length)

    def _encode_target(self, target: str, max_length: int) -> List[int]:
        eos = self.tokenizer.eos_token or ""
        return self._encode_text(target + eos, max_length=max_length)

    def _prompt_limit(self, has_target: bool) -> int:
        if self.max_text_length is None:
            return self.max_prompt_length

        if has_target:
            return max(1, min(self.max_prompt_length, self.max_text_length - 1))

        return max(1, min(self.max_prompt_length, self.max_text_length))

    def encode(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if targets is not None and len(prompts) != len(targets):
            raise ValueError("prompts and targets must have the same length.")

        rows = []
        label_rows = []
        prompt_lengths = []
        has_target = targets is not None

        for idx, prompt in enumerate(prompts):
            prompt_ids = self._encode_prompt(
                prompt,
                max_length=self._prompt_limit(has_target),
            )
            prompt_lengths.append(len(prompt_ids))

            if targets is None:
                input_ids = prompt_ids
                labels = None
            else:
                if self.max_text_length is None:
                    target_limit = self.max_target_length
                else:
                    remaining = self.max_text_length - len(prompt_ids)
                    target_limit = max(1, min(self.max_target_length, remaining))

                target_ids = self._encode_target(
                    targets[idx],
                    max_length=target_limit,
                )
                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids

            rows.append(input_ids)
            label_rows.append(labels)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id.")

        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        padded_labels = []

        for row, labels in zip(rows, label_rows):
            pad_len = max_len - len(row)
            padded.append(row + [pad_id] * pad_len)
            masks.append([1] * len(row) + [0] * pad_len)

            if targets is not None:
                padded_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "prompt_lengths": torch.tensor(
                prompt_lengths,
                dtype=torch.long,
                device=device,
            ),
        }

        if targets is not None:
            batch["labels"] = torch.tensor(
                padded_labels,
                dtype=torch.long,
                device=device,
            )

        return batch
