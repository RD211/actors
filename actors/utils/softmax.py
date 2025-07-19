import torch
import torch.nn.functional as F


def _selective_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Taken from: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1775
    per_token_logps = []
    for row_logits, row_labels in zip(
        logits, index, strict=False
    ):  # loop to reduce peak mem consumption
        row_logps = F.log_softmax(row_logits, dim=-1)
        row_per_token_logps = row_logps.gather(
            dim=-1, index=row_labels.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps.append(row_per_token_logps)
    per_token_logps = torch.stack(per_token_logps)
    return per_token_logps
