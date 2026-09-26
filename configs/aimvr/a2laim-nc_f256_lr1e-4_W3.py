# AIM-VR (ICME 2025) training config: A2LAimNet, 256 features, W3 split of ALL50
#
# Released training config of the AIM-VR model (AimVR wrapper + A2LAimNet
# generator, num_features=256). Trained on Train_W3 of the ALL50 dataset and
# validated on RainMotion (rain), KITTI_snow (snow) and REVIDE_indoor (haze).
# Model parameters follow the evidence-locked experiment config
# a2laim-nc_f256_lr1e-4_W3_20241211.py.
default_scope = 'mmagic'  # default registry scope for module lookup; otherwise mmengine raises errors

load_batch_size = 1
load_input_frames = 5
input_val_frames = 10
# input_frames = 2 * load_input_frames  # commented-out option: double the frame count
input_frames = load_input_frames  # use the original frame count directly
iter_k = 300
iters = 1000 * iter_k
interval_val = 5000

experiment_name = (
    f'a2laim-l2nc_f256_4x{input_frames}xb{load_batch_size}-lr1e-4-{iter_k}k_W3_20241211'  # single-frequency enhancement restriction
)
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 1   # video restoration mode: deraining, dehazing and desnowing

# checkpoint path - passed in via the command line (--cfg-options load_from=...)
load_from = None
resume = False


# model settings
model = dict(
    type='AimVR',
    generator=dict(
        type='A2LAimNet',
        num_features=256,
        scale_factor=scale,
        feat_pretrained='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
    ),
    num_input_frames=load_input_frames,
    scale_factor=scale,
    train_cfg=dict(),
    test_cfg=dict(
        scale_factor=scale,
        window_size=[2, int(8 / scale), int(8 / scale)],  # temporal and spatial window size
        tile=[
            load_input_frames,  # if 0, feed the full-length video
            int(256 / scale),   # if 0, feed the full-resolution frame
            int(256 / scale),
        ],  # [temporal segment size, spatial patch height, spatial patch width]
        tile_overlap=[
            load_input_frames - 1,
            int(192 / scale),
            int(192 / scale),
        ],  # [temporal overlap, spatial height overlap, spatial width overlap]
        use_temporal_gradient=True,  # whether to use Gaussian weights along time
        use_temporal_average=False,  # whether to use cumulative averaging along time
        use_spatial_gradient=True,  # whether to use Gaussian weights spatially
        use_spatial_average=False,  # whether to keep cumulative averaging spatially
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),     # pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    # consistent with the derainer: use vgg16
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '3': 1.0,    # relu1_2
            '8': 1.0,    # relu2_2
            '15': 0.5,   # relu3_3
        },
        vgg_type='vgg16',
        criterion='mse',
        perceptual_weight=0.02,
        style_weight=0,
        norm_img=False,
        pretrained='torchvision://vgg16'  # vgg_type is specified here
    ),
    # # consistent with RealBasicVSR: use vgg19
    # perceptual_loss=dict(
    #     type='PerceptualLoss',
    #     layer_weights={
    #         '2': 0.1,
    #         '7': 0.1,
    #         '16': 1.0,
    #         '25': 1.0,
    #         '34': 1.0,
    #     },
    #     vgg_type='vgg19',
    #     perceptual_weight=1.0,
    #     style_weight=0,
    #     norm_img=False),
    # # use the default PerceptualLoss
    # # perceptual_loss=dict(type='PerceptualLoss', loss_weight=0.3, reduction='mean'),
    # contrast_loss=dict(type='ContrastLoss', loss_weight=0.05, reduction='mean'),
    ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ),
)

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:05d}.jpg'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    # dict(type='FixedCrop', keys=['img', 'gt'], crop_size=(256, 256)),  # downscale during training
    dict(type='PairedRandomCrop', gt_patch_size=256),                                   # use BI x4 directly
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    # dict(type='MirrorSequence', keys=['img', 'gt']),  # extend the sequence (x1, ..., xN, xN, ..., x1). # commented out: mirror sequence flip
    # UnsharpMasking: sharpen the GT frames to enhance fine details
    # dict(
    #     type='UnsharpMasking',
    #     keys=['gt'],
    #     kernel_size=51,
    #     sigma=0,
    #     weight=0.5,
    #     threshold=10),

    dict(
        type='ColorJitter',   # pixel-level data augmentation
        keys=['img', 'gt'],
        channel_order='rgb',
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05),
    dict(type='Clip', keys=['img']),  # clamp pixel values to 0-1, preventing degraded img from going out of range
    dict(type='PackInputs'),
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs'),
]

data_root = 'data/ALL50'  # TODO: replace with your local ALL50 dataset root

train_dataloader = dict(
    num_workers=10,
    batch_size=load_batch_size,
    drop_last=True,     # whether to drop the last batch
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='Train_W3', task_name='vr'),
        data_root=f'{data_root}/Train_W3',
        data_prefix=dict(img='IMG', gt='GT'),
        depth=1,
        num_input_frames=load_input_frames,
        # fixed_seq_len=100,
        pipeline=train_pipeline,
    ),
)

val_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:05d}.jpg',
    ),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs'),
]

# validation set
data_val = f'{data_root}/Test_W3'
# test set
data_test = f'{data_root}/Test_W3'

# RainMotion validation and test sets
RainMotion_val_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainMotion', task_name='vr'),
        data_root=f'{data_val}/RainMotion',
        data_prefix=dict(img='IMG', gt='GT'),
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)
RainMotion_test_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='RainMotion', task_name='vr'),
        data_root=f'{data_test}/RainMotion',
        data_prefix=dict(img='IMG', gt='GT'),
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)
RainMotion_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='RainMotion'),
        dict(type='SSIM', convert_to='Y', prefix='RainMotion'),
        # dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='RainMotion'),
    ],
)

# KITTI (snow) test set configuration (same structure as Rain, with adapted paths and prefixes)
KITTI_val_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='KITTI', task_name='vr'),
        data_root=f'{data_val}/KITTI_snow',
        data_prefix=dict(img='IMG', gt='GT'),
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)
KITTI_test_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='KITTI', task_name='vr'),
        data_root=f'{data_test}/KITTI_snow',
        data_prefix=dict(img='IMG', gt='GT'),
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)
KITTI_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='KITTI'),
        dict(type='SSIM', convert_to='Y', prefix='KITTI'),
        # dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='KITTI'),
    ],
)

# Fog (REVIDE) test set configuration (same structure as Rain, with adapted paths and prefixes)
REVIDE_val_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='REVIDE', task_name='vr'),
        data_root=f'{data_val}/REVIDE_indoor',
        data_prefix=dict(img='IMG', gt='GT'),
        # num_input_frames=input_val_frames,
        num_input_frames=input_frames,  # REVIDE frames are large; use fewer frames to save resources
        pipeline=val_pipeline,
    ),
)
REVIDE_test_dataloader = dict(
    num_workers=10,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='REVIDE', task_name='vr'),
        data_root=f'{data_test}/REVIDE_indoor',
        data_prefix=dict(img='IMG', gt='GT'),
        num_input_frames=input_val_frames,
        pipeline=val_pipeline,
    ),
)
REVIDE_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y', prefix='REVIDE'),
        dict(type='SSIM', convert_to='Y', prefix='REVIDE'),
        # dict(type='NIQE', input_order='CHW', convert_to='Y', prefix='REVIDE'),
    ],
)

# multi-dataset validation and test configuration
val_dataloader = [
    RainMotion_val_dataloader,
    KITTI_val_dataloader,
    REVIDE_val_dataloader,
]
val_evaluator = [
    RainMotion_evaluator,
    KITTI_evaluator,
    REVIDE_evaluator,
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=interval_val)    # validation interval
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    # accumulative_counts=5,                                              # gradient accumulation
    # optimizer=dict(type='Adam', lr=4e-4, betas=(0.9, 0.99)),
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001),
    # paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}),
    clip_grad=dict(max_norm=4.0, norm_type=2),                          # gradient clipping by norm
)

# model compilation to speed up training
cfg = dict(compile='compile_options')

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50000,
    ),
    dict(
        type='CosineRestartLR',
        periods=[(iters-150000)],
        restart_weights=[1],
        by_epoch=False,
        begin=150000,
        end=iters,
        eta_min=1e-7
    )
]


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval_val,  # validation interval
        save_optimizer=True,  # save optimizer state; changing the optimizer later may prevent resuming
        out_dir=save_dir,
        max_keep_ckpts=20,
        save_best=[
            'RainMotion/PSNR',
            'KITTI/PSNR',
            'REVIDE/PSNR',
        ],
        rule=['greater', 'greater', 'greater'],
        by_epoch=False,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=True,  # let cuDNN select the fastest convolution algorithm (slower startup); default False
    mp_cfg=dict(
        mp_start_method='fork', opencv_num_threads=4
    ),  # fork (Linux); spawn (Windows); 'forkserver' may perform better in some cases
    dist_cfg=dict(backend='nccl'),  # use the gloo backend for distributed training on Windows
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)


# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # Optional: enable Weights & Biases logging by uncommenting below and
    # setting your own project/run name (requires `pip install wandb` and login).
    # dict(
    #     type='WandbVisBackend',
    #     init_kwargs=dict(
    #         project='mmagic',
    #         name='a2l256nc_4x1x5_vr'
    #     ),
    #     save_dir=work_dir
    # )
]  # local logging by default; enable WandbVisBackend above if needed

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['input', 'pred_img', 'gt_img'],
    bgr2rgb=True,
)

# visualization hooks
custom_hooks = [dict(type='BasicVisualizationHook', interval=10)]

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True,
)  # distributed training related
