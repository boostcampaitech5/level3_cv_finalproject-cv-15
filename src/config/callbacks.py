from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from src.config import full_builds

ModelCheckpointConfig = full_builds(
    ModelCheckpoint,
    dirpath="checkpoint/${logger.name}",
    filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
    save_last=True,
    monitor="val_loss",
    save_top_k=2,
    mode="min",
    save_weights_only=True,
    auto_insert_metric_name=False,
)

LearningRateMonitorConfig = full_builds(
    LearningRateMonitor,
    logging_interval="step",
)

EarlyStoppingConfig = full_builds(
    EarlyStopping,
    monitor="val_loss",
    patience=3,
    mode="min",
    strict=True,
    check_finite=True,
)

RichModelSummaryConfig = full_builds(RichModelSummary, max_depth=3)

RichProgressBarThemeConfig = full_builds(
    RichProgressBarTheme,
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
)

RichProgressBarConfig = full_builds(RichProgressBar, theme=RichProgressBarThemeConfig)

BasicCallbackConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

NoEarlyStopCallbackConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

NoCheckPointConfig = builds(
    list,
    [
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

NoEalryCheckPointConfig = builds(
    list,
    [
        LearningRateMonitorConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

OnlyCheckPointConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
    ],
)

SkinModelCheckpointConfig = full_builds(
    ModelCheckpoint,
    dirpath="checkpoint/${logger.name}",
    filename="epoch={epoch:02d}-dice_score={dice_score:.2f}",
    save_last=True,
    monitor="dice_score",
    save_top_k=2,
    mode="max",
    save_weights_only=True,
    auto_insert_metric_name=False,
)

SkinEarlyStoppingConfig = full_builds(
    EarlyStopping,
    monitor="dice_score",
    patience=3,
    mode="max",
    strict=True,
    check_finite=True,
)

SkinCallbackConfig = builds(
    list,
    [
        SkinModelCheckpointConfig,
        LearningRateMonitorConfig,
        SkinEarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="callbacks", name="basic_callbacks", node=BasicCallbackConfig)
    cs.store(group="callbacks", name="no_ealrystop", node=NoEarlyStopCallbackConfig)
    cs.store(group="callbacks", name="no_checkpoint", node=NoCheckPointConfig)
    cs.store(
        group="callbacks", name="no_ealry_no_ckeckpoint", node=NoEalryCheckPointConfig
    )
    cs.store(group="callbacks", name="only_checkpoint", node=OnlyCheckPointConfig)
    cs.store(group="callbacks", name="skin_callbacks", node=SkinCallbackConfig)
