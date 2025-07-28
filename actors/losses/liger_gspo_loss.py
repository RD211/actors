from typing import TYPE_CHECKING

from actors.losses.liger_grpo_loss import AllowedLoss, LigerGRPOLoss
from actors.utils.gspo_loss import LigerFusedLinearGSPOLoss

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg


class LigerGSPOLoss(LigerGRPOLoss):
    def __init__(
        self,
        config: "ActorTrainCfg",
        loss_type: AllowedLoss = "grpo",
    ) -> None:
        super().__init__(config=config, loss_type=loss_type)

        if loss_type not in ("grpo", "bnpo", "dr_grpo"):
            raise ValueError(f"invalid loss_type '{loss_type}'")

        self.loss: LigerFusedLinearGSPOLoss = LigerFusedLinearGSPOLoss(
            beta=self.beta,
            use_ref_model=self.beta > 0.0,
            loss_type=loss_type,
            temperature=self.temperature,
        )
