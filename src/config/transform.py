from albumentations import Compose, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from hydra.core.config_store import ConfigStore
from hydra_zen import builds

from src.config import full_builds

ResizeConfig = full_builds(Resize, height=512, width=512, always_apply=True)

NormalizeConfig = full_builds(
    Normalize,
    always_apply=True,
)

ToTensorV2Config = full_builds(ToTensorV2, always_apply=True)

HorizontalFlipConfig = full_builds(HorizontalFlip, p=0.5)


BasicConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

SkinConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig
        ],
    ),
)

def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="transforms",
        name="basic",
        node=BasicConfig,
    )
    cs.store(
        group="transforms",
        name="skin",
        node=SkinConfig,
    )

