from hydra.core.config_store import ConfigStore
from torch.nn import BCEWithLogitsLoss
from src.loss import DiceBCELoss

from src.config import full_builds

BCEWithLogitsLossConfig = full_builds(BCEWithLogitsLoss)

DiceBCELossConfig = full_builds(DiceBCELoss, dice_smooth=1.0, bce_weight=0.5)

def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="loss", name="bce_with_logits", node=BCEWithLogitsLossConfig)
    cs.store(group="loss", name="dice_bce", node=DiceBCELossConfig)
