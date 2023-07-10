from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from src.config import partial_builds
from src.data import EyesDataSet, MNISTDataSet, CatSkinDataset

MNISTTrainDatasetConfig = partial_builds(
    MNISTDataSet,
    root="./data",
    train=True,
    transform=None,
    download=True,
)

MNISTValidationDatasetConfig = partial_builds(
    MNISTDataSet,
    root="./data",
    train=True,
    transform=None,
    download=True,
)

MNISTTrainDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=64,
    shuffle=True,
    num_workers=12,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

MNISTValidationDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=64,
    num_workers=12,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

EyesTrainDatasetConfig = partial_builds(
    EyesDataSet,
    path="./data/eyes/dog_ophthalmoscope_train.json",
    root="./data/eyes/dog",
    transform=None,
)

EyesValidationDatasetConfig = partial_builds(
    EyesDataSet,
    path="./data/eyes/dog_ophthalmoscope_val.json",
    root="./data/eyes/dog",
    transform=None,
)

EyesTrainDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
)

EyesValidationDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=32,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
)

CatSkinTrainDatasetConfig = partial_builds(
    CatSkinDataset,
    is_train=True,
    transform=None,
)

CatSkinValidDatasetConfig = partial_builds(
    CatSkinDataset,
    is_train=False,
    transform=None,
)

CatSkinTrainDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=8,
    shuffle=True,
    num_workers=7,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

CatSkinValidDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=8,
    num_workers=7,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="train_dataset",
        name="basic_train_dataset",
        node=MNISTTrainDatasetConfig,
    )
    cs.store(
        group="val_dataset",
        name="basic_val_dataset",
        node=MNISTValidationDatasetConfig,
    )

    cs.store(
        group="train_loader",
        name="basic_train_loader",
        node=MNISTTrainDataloaderConfig,
    )
    cs.store(
        group="val_loader",
        name="basic_val_loader",
        node=MNISTValidationDataloaderConfig,
    )

    cs.store(
        group="train_dataset",
        name="eyes_train_dataset",
        node=EyesTrainDatasetConfig,
    )
    cs.store(
        group="val_dataset",
        name="eyes_val_dataset",
        node=EyesValidationDatasetConfig,
    )

    cs.store(
        group="train_loader",
        name="eyes_train_loader",
        node=EyesTrainDataloaderConfig,
    )
    cs.store(
        group="val_loader",
        name="eyes_val_loader",
        node=EyesValidationDataloaderConfig,
    )

    cs.store(
        group="train_dataset",
        name="catskin_train_dataset",
        node=CatSkinTrainDatasetConfig,
    )
    
    cs.store(
        group="val_dataset",
        name="catskin_val_dataset",
        node=CatSkinValidDatasetConfig,
    )

    cs.store(
        group="train_loader",
        name="catskin_train_loader",
        node=CatSkinTrainDataloaderConfig,
    )
    
    cs.store(
        group="val_loader",
        name="catskin_val_loader",
        node=CatSkinValidDataloaderConfig,
    )