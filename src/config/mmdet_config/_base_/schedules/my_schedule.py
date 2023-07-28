max_epochs = 80
num_last_epochs = 10

# training schedule 
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# param_scheduler = [
#     dict(
#         # use quadratic formula to warm up 1 epochs
#         type='QuadraticWarmupLR',
#         by_epoch=True,
#         begin=0,
#         end=1,
#         convert_to_iter_based=True),
#     dict(
#         # use cosine lr from 1 to 70 epoch
#         type='CosineAnnealingLR',
#         eta_min=0.00001 * 0.05,
#         begin=1,
#         T_max=max_epochs - num_last_epochs,
#         end=max_epochs - num_last_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         # use fixed lr during last 10 epochs
#         type='ConstantLR',
#         by_epoch=True,
#         factor=1,
#         begin=max_epochs - num_last_epochs,
#         end=max_epochs,
#     )
# ]

# optimizer

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
