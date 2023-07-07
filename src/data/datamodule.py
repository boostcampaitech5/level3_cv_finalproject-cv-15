from lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        transforms,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.transforms = transforms

    def train_dataloader(self):
        return self.train_loader(dataset=self.train_dataset(transform=self.transforms))

    def val_dataloader(self):
        return self.val_loader(dataset=self.val_dataset(transform=self.transforms))
