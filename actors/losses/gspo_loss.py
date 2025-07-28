from typing import TYPE_CHECKING

from actors.losses.grpo_loss import AllowedLoss, GRPOLoss

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg


class GSPOLoss(GRPOLoss):
    def __init__(
        self,
        config: "ActorTrainCfg",
        eps_low: float = 0.2,
        eps_high: float = 0.2,
        loss_type: AllowedLoss = "grpo",
        delta: float | None = None,
        max_completion_length: int | None = None,
    ):
        super().__init__(
            config=config,
            eps_low=eps_low,
            eps_high=eps_high,
            loss_type=loss_type,
            delta=delta,
            max_completion_length=max_completion_length,
            gspo=True,  # GSPOLoss is a variant of GRPOLoss with gspo=True
        )
