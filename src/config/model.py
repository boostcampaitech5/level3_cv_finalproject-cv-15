from hydra.core.config_store import ConfigStore
from timm import create_model

from src.config import full_builds
from src.models import ResNet

TimmCreateModelConfig = full_builds(
    create_model, pretrained=True, num_classes="${other.num_classes}"
)

ResNetConfig = full_builds(
    ResNet,
    in_channels=1,
    start_channels=64,
    class_num="${other.num_classes}",
    blocks=[2, 2, 2, 2],
)

TimmResNset50DConfig = TimmCreateModelConfig(model_name="resnest50d.in1k")
TimmResNset101EConfig = TimmCreateModelConfig(model_name="resnest101e.in1k")
TimmResNset200EConfig = TimmCreateModelConfig(model_name="resnest200e.in1k")
TimmResNset269EConfig = TimmCreateModelConfig(model_name="resnest269e.in1k")


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="resnet",
        node=ResNetConfig,
    )
    cs.store(
        group="architecture",
        name="timm_resnest50d",
        node=TimmResNset50DConfig,
    )
    cs.store(
        group="architecture",
        name="timm_resnest101e",
        node=TimmResNset101EConfig,
    )
    cs.store(
        group="architecture",
        name="timm_resnest200e",
        node=TimmResNset200EConfig,
    )
    cs.store(
        group="architecture",
        name="timm_resnest269e",
        node=TimmResNset269EConfig,
    )
