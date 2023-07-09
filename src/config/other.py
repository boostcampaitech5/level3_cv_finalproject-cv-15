from dataclasses import dataclass
from typing import Union

from hydra.core.config_store import ConfigStore

from src.config import full_builds


@dataclass
class Other:
    load_ckpt_path: Union[str, None]
    num_classes: Union[int, None]


OtherConfig = full_builds(Other, load_ckpt_path=None, num_classes=11)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="other", name="basic", node=OtherConfig)
