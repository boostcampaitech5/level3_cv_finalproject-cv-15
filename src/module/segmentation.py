from typing import Any
from typing import Union
import torch
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SegmentationModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        scheduler: LRScheduler,
        load_ckpt_path: Union[str, None] = None,
        num_classes: Union[int, None] = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.dice_score = []

    def dice_coef(self, y_true, y_pred):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2.0 * intersection + eps) / (
            torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)

        self.log(
            name="train_loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.dice_score = []
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        output = self.forward(x)
        output = (output > 0.5).detach().cpu()
        y = y.detach().cpu()
        loss = self.dice_coef(output, y)
        self.dice_score.append(loss)

        return loss
        
    def on_validation_epoch_end(self) -> None:
        dices = torch.cat(self.dice_score, 0)
        dice_per_class = torch.mean(dices, 0)
        avg_dice = torch.mean(dice_per_class).item()
        
        for class_idx, score in enumerate(dice_per_class):
            self.log(
                name="%d_class_dice_score" % class_idx,
                value=score,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        
        self.log(
            name="dice_score",
            value=avg_dice,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return avg_dice

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
