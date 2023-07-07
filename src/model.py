from typing import Union

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class Model(LightningModule):
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
        self.example_input_array = torch.zeros(1, 3, 400, 400)
        self.val_loss = MeanMetric()
        self.num_classes = num_classes
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average=None)
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average=None)

        if load_ckpt_path is not None:
            ckpt = torch.load(load_ckpt_path)
            state_dict = {
                k.partition("model.")[2]: v for k, v in ckpt["state_dict"].items()
            }
            self.model.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)

        self.train_acc.update(pred.argmax(dim=-1), y.argmax(-1))

        self.log(
            name="train_loss",
            value=round(loss.item(), 4),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        print(f"train_acc : {self.train_acc.compute()}")
        self.train_acc.reset()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.val_loss(loss)

        self.val_acc.update(pred.argmax(dim=-1), y.argmax(-1))

        self.log(
            name="val_loss",
            value=self.val_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_validation_epoch_end(self) -> None:
        print(f"val_acc : {self.val_acc.compute()}")
        self.val_acc.reset()

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
