from hydra.core.config_store import ConfigStore

from src.config import partial_builds
from src.module import ClassificationModel, SegmentationModel

ClassificationConfig = partial_builds(ClassificationModel)
SegmentationConfig = partial_builds(SegmentationModel)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="module", name="classification", node=ClassificationConfig)
    cs.store(group="module", name="segmentation", node=SegmentationConfig)
