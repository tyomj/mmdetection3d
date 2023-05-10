_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(CLASSES=class_names)
backend_args = None
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_paddings=((0, 0, 0), ((1, 1, 1), 0, 0), ((1, 1, 1), 0, 0),
                          ((0, 1, 1), 0, 0)),
        return_middle_feats=True),
    points_encoder=dict(
        type='VoxelSetAbstraction',
        keypoints_sampler=dict(
            type='SPCSampler',
            num_keypoints=4096,
            num_sectors=6,
            roi_neighbor_radius=1.6),
        fused_out_channel=90,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(
                type='VectorPoolAggregationModuleMSG',
                source_feats_index=2,
                scale_factor=4,
                in_channels=64,
                mlp_channels=[[128]],
                local_aggregation_type='local_interpolation',
                num_aggregation_channels=32,
                neighbor_distance_multiplier=1.0,
                filter_neighbor_with_roi=True,
                roi_neighbor_radius=4.0,
                num_reduced_channels=32,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=-1,
                        max_neighbor_distance=1.2,
                    ),
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=-1,
                        max_neighbor_distance=2.4,
                    ),
                ]),
            dict(
                type='VectorPoolAggregationModuleMSG',
                source_feats_index=3,
                scale_factor=8,
                in_channels=64,
                mlp_channels=[[128]],
                local_aggregation_type='local_interpolation',
                num_aggregation_channels=32,
                neighbor_distance_multiplier=1.0,
                filter_neighbor_with_roi=True,
                roi_neighbor_radius=6.4,
                num_reduced_channels=32,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=-1,
                        max_neighbor_distance=2.4,
                    ),
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=-1,
                        max_neighbor_distance=4.8,
                    ),
                ]),
        ],
        rawpoints_sa_cfgs=dict(
            type='VectorPoolAggregationModuleMSG',
            in_channels=32,
            mlp_channels=[[32]],
            local_aggregation_type='local_interpolation',
            num_aggregation_channels=32,
            neighbor_distance_multiplier=1.0,
            filter_neighbor_with_roi=True,
            roi_neighbor_radius=2.4,
            num_reduced_channels=1,
            groups_cfg_list=[
                dict(
                    num_local_voxel=(2, 2, 2),
                    post_mlps=(32, 32),
                    neighbor_nsample=-1,
                    max_neighbor_distance=0.2,
                ),
                dict(
                    num_local_voxel=(3, 3, 3),
                    post_mlps=(32, 32),
                    neighbor_nsample=-1,
                    max_neighbor_distance=0.4,
                ),
            ]),
        bev_feat_channel=256,
        bev_scale_factor=8),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['Vehicle']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),  # center_z ?
            dim=(3, 2),
            rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size,  # [:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    roi_head=dict(
        type='PVRCNNRoiHead',
        num_classes=3,
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=544,
            extra_width=0.1,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                activated=True,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(
                type='VectorPoolAggregationModuleMSG',
                in_channels=64,
                mlp_channels=[[128]],
                local_aggregation_type='voxel_random_choice',
                num_aggregation_channels=32,
                neighbor_distance_multiplier=1.0,
                filter_neighbor_with_roi=True,
                num_reduced_channels=30,
                groups_cfg_list=[
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=32,
                        max_neighbor_distance=0.8,
                    ),
                    dict(
                        num_local_voxel=(3, 3, 3),
                        post_mlps=(64, 64),
                        neighbor_nsample=32,
                        max_neighbor_distance=1.6,
                    ),
                ]),
        ),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes=3,
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False,
            grid_size=[704, 800, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0.1,
            use_rotate_nms=True,
            nms_type='rotate',
            post_center_limit_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            score_threshold=0.1,
            pre_max_size=1000,
            post_max_size=83,
        ),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.1,
            score_thr=0.1)))
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
eval_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr))
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=lr * 10,
        begin=0,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        eta_min=lr * 1e-4,
        begin=15,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=15,
        eta_min=0.85 / 0.95,
        begin=0,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=25,
        eta_min=1,
        begin=15,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]
