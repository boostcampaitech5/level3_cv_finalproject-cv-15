from hydra_zen import instantiate
from omegaconf import OmegaConf

from src.data import DataModule
from src.utils import set_seed


def train(config):
    exp = instantiate(config)

    set_seed(42)

    architecture = exp.architecture
    optimizer = exp.optimizer
    loss = exp.loss
    scheduler = exp.scheduler
    trainer = exp.trainer(logger=exp.logger, callbacks=exp.callbacks)

    model = exp.module(
        model=architecture,
        optimizer=optimizer,
        loss=loss,
        scheduler=scheduler,
        load_ckpt_path=exp.other.load_ckpt_path,
        num_classes=exp.other.num_classes,
    )

    datamodule = DataModule(
        train_dataset=exp.train_dataset,
        train_loader=exp.train_loader,
        val_dataset=exp.val_dataset,
        val_loader=exp.val_loader,
        transforms=exp.transforms,
    )

    trainer.logger.watch(
        model=model,
        log="gradients",
        log_freq=10,
    )

    trainer.logger.experiment.config.update(OmegaConf.to_container(config))

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
