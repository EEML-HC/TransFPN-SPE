_base_ = [
    '../../_base_/datasets/voc0712.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
pretrained = 'mae_pretrain_vit_base_full.pth'
load_from  = 'work_dirs/voc_base_split2/latest.pth'
norm_cfg = dict(type='LN', requires_grad=True)
data_root = 'data/datasets/'
dataset_type = 'VOCDataset'

model = dict(
    type='imTED',
    pretrained=pretrained,
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.2,
        learnable_pos_embed=True,
        use_checkpoint=True,
        with_simple_fpn=True,
        last_feat=True),
    neck=dict(
        type='SimpleFPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        norm_cfg=norm_cfg,
        use_residual=False,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_skip_fpn=False,
    with_mfm=True,
    roi_head=dict(
        type='imTEDRoIHead',
        bbox_roi_extractor=[dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=768,
            featmap_strides=[4, 8, 16, 32]),
                            dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=768,
            featmap_strides=[16])],
        bbox_head=dict(
            type='MAEBBoxHead',
            use_checkpoint=False,
            in_channels=768,
            img_size=224,
            patch_size=16, 
            embed_dim=512, 
            depth=8,
            num_heads=16, 
            mlp_ratio=4., 
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale=(640,640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='RandomFlip', flip_ratio=0.5), 
    #dict(type='Expand', ratio_range=(1,3), seg_ignore_label=None, prob=0.3), 
 
    #dict(type='Corrupt', corruption='gaussian_noise', severity=1), 
    #dict(type='Corrupt', corruption='snow', severity=1), 
    #dict(type='Corrupt', corruption='frost', severity=1), 
    #dict(type='Corrupt', corruption='fog', severity=1), 
    #dict(type='Corrupt', corruption='brightness', severity=1), 

    dict(type='Mosaic',
	 img_scale=img_scale,
	 pad_val=114.0), 
    dict(type='RandomAffine',
         scaling_ratio_range=(0.1, 2),
         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='PackSegInputs')
    #dict(type='MixUp', 
         #img_scale=img_scale, 
         #ratio_range=(0.8, 1.6), 
         #pad_val=114.0)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

evaluation = dict(interval=6, metric='mAP')
checkpoint_config = dict(interval=6)
classes =  ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                         'chair', 'diningtable', 'dog', 'motorbike', 'person',
                         'pottedplant', 'sheep', 'train', 'tvmonitor','Arrester','Circuit-breaker','Current transformer','Voltage transformer','Isolating switch')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        #type='RepeatDataset',
	times=1,
	pipeline=train_pipeline,
	type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2012/ImageSets/Main/trainsplit110shots.txt'
            ],
            img_prefix=[data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructorBackboneFronzen', 
    paramwise_cfg=dict(
            num_encoder_layers=12, 
            num_decoder_layers=8, 
            layer_decay_rate=0.8,
    )
)
# learning policy
lr_config = dict(policy='step', step=[81, 99])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=42)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
# find_unused_parameters=True
