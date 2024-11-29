_base_ = [
    'swin-large-patch4-window7-in22k-pre_upernet_'
    '8xb2-160k_ade20k-512x512.py',
]
# checkpoint_config = dict(interval=50)
data_root = "/data/ephemeral/home/data"

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.,0.,0.],
    std=[255.,255.,255.],
    bgr_to_rgb=True,
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
)

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
norm_cfg = None
model = dict(
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[0.,0.,0.],
        std=[255.,255.,255.],
        bgr_to_rgb=True,
        size=(1024, 1024),
        pad_val=0,
        seg_pad_val=255),
    
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        window_size=12),
    decode_head=dict(in_channels=[192, 384, 768, 1536], 
                     num_classes=29, 
                     norm_cfg=norm_cfg,
                     loss_decode=dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=True,
                         loss_weight=1.0,)),
    auxiliary_head=dict(in_channels=768, 
                        num_classes=29,
                        norm_cfg=norm_cfg,
                        loss_decode=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=True,
                            loss_weight=0.4,
                        )))


# dataset settings
dataset_type = 'XRayDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayAnnotations'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='LoadXRayAnnotations'),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='PackSegInputs')
]

metainfo = {
    'classes': [
      'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
      'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
      'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
      'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
      'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
      'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ],
    'palette': [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    ]
}

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        mode='train',
        metainfo=metainfo,
        pipeline=train_pipeline,
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        mode='train',
        metainfo=metainfo,
        pipeline=val_pipeline,
        ))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        mode='test',
        metainfo=metainfo,
        pipeline=test_pipeline,
        ))

val_evaluator = dict(type='DiceMetric')
test_evaluator = dict(type='RLEncoding')
  

################## Wandb 연결 ####################
visualizer = dict(
    type='SegLocalVisualizer',    
    vis_backends = [
        dict(type='LocalVisBackend'), 
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 entity="lv2-ss-",
                 project="semantic-segmentation",
                 name='38.1_swin-l_fold3'))],
    name='visualizer'    
    )
#################################################