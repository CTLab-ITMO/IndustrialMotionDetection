from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from hiera import hiera
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

from src.models.box_list import BoxList
from src.models.image_list import ImageList
from AlphAction.alphaction.modeling.poolers import make_3d_pooler

import decord

decord.bridge.set_bridge('torch')


class ROIPoolingCfg:
    POOLER_RESOLUTION: int = 7
    POOLER_SCALE: float = 0.0625
    POOLER_SAMPLING_RATIO: int = 0
    POOLER_TYPE: str = 'align3d'
    MEAN_BEFORE_POOLER: bool = True


import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads


class WeightedRoIHeads(RoIHeads):
    def __init__(self, *args, class_weights=None, use_focal_loss=False, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha = alpha

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        if self.use_focal_loss:
            probs = F.softmax(class_logits, dim=1)

            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)

            pt = (one_hot * probs).sum(1)
            focal_weight = (1 - pt) ** self.gamma

            alpha_weight = torch.ones_like(labels).float()
            alpha_weight[labels > 0] = self.alpha
            alpha_weight[labels == 0] = 1 - self.alpha

            if self.class_weights is not None:
                class_weight = self.class_weights[labels]
                focal_weight = focal_weight * class_weight

            focal_weight = focal_weight * alpha_weight

            ce_loss = F.cross_entropy(class_logits, labels, reduction='none')
            classification_loss = (focal_weight * ce_loss).mean()
        else:
            if self.class_weights is not None:
                classification_loss = F.cross_entropy(class_logits, labels, weight=self.class_weights)
            else:
                classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


class HieraActionDetector(nn.Module):
    def __init__(self, num_classes: int, class_weights=None, use_focal_loss=False, gamma=2, alpha=1, temp_kernel=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = hiera.hiera_base_16x224(num_classes=num_classes,
                                                pretrained=True,
                                                checkpoint="mae_k400")
        cfg = self.backbone.config
        self.test_ext = (0.1, 0.05)
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dims = [
            cfg["embed_dim"],
            cfg["embed_dim"] * 2,
            cfg["embed_dim"] * 4,
            cfg["embed_dim"] * 8,
        ]
        embed_dim = 384
        self.temporal_convs = nn.ModuleList([
            nn.Conv3d(
                in_channels=ch,
                out_channels=384,
                kernel_size=(temp_kernel, 1, 1),
                padding=(temp_kernel // 2, 0, 0),
            )
            for ch in embed_dims
        ])

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[embed_dim, embed_dim, embed_dim, embed_dim],
            out_channels=embed_dim
        )

        def generate_anchor():
            base_sizes = [16, 32, 64, 128]
            scales = [2 ** (-0.5), 1.0, 2 ** (1 / 4), 2 ** (2 / 4), 2 ** (3 / 4)]
            ratios = [0.2, 0.33, 0.5, 1.0, 2]
            anchor_sizes = tuple(
                tuple(size * scale for scale in scales)
                for size in base_sizes
            )
            aspect_ratios = tuple(
                tuple(ratios)
                for _ in base_sizes
            )

            print("anchor_sizes:", anchor_sizes)
            print("aspect_ratios:", aspect_ratios)

            return AnchorGenerator(anchor_sizes, aspect_ratios)

        out_channels = embed_dim

        rpn_anchor_generator = generate_anchor()
        rpn_head = torchvision.models.detection.rpn.RPNHead(
            out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0]
        )

        # number of proposals to keep after applying NMS.
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # NMS threshold used for postprocessing the RPN proposals
        rpn_nms_thresh = 0.7

        # minimum IoU between the anchor and the GT box so that they can be
        # considered as positive during training of the RPN.
        rpn_fg_iou_thresh = 0.7

        # maximum IoU between the anchor and the GT box so that they can be
        # considered as negative during training of the RPN.
        rpn_bg_iou_thresh = 0.3

        # --- BalancedPositiveNegativeSampler ---

        # number of anchors that are sampled during training of the RPN for computing the loss
        rpn_batch_size_per_image = 256

        # proportion of positive anchors in a mini-batch during training of the RPN
        rpn_positive_fraction = 0.5
        # ---------------------------------------

        # only return proposals with an objectness score greater than score_thresh
        rpn_score_thresh = 0.0

        self.rpn_network = torchvision.models.detection.rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=7,
            sampling_ratio=2)

        representation_size = 1024
        resolution = box_roi_pool.output_size[0]
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            representation_size,
            num_classes)

        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5

        box_batch_size_per_image = 512
        box_positive_fraction = 0.25

        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100

        self.roi_heads = WeightedRoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,

            #training
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,

            #inference
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,

            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            gamma=gamma,
            alpha=alpha,
        )

        self.bbox_head = nn.Linear(embed_dim, num_classes * 4)
        # ---

        # rois setting
        self.head_cfg = ROIPoolingCfg()
        self.pooler = make_3d_pooler(self.head_cfg)
        resolution = self.head_cfg.POOLER_RESOLUTION
        self.max_pooler = nn.MaxPool2d((resolution, resolution))

        self.proposal_per_clip = 300000

    def forward(self, video, boxes, labels):
        if self.training:
            input_proposals = boxes
        else:
            input_proposals = [box.extend(self.test_ext) for box in boxes]

        images = video[:, :, video.shape[2] // 2]

        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input should have last two elements H and W, got {image_size}"
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = input_proposals[i].bbox
            d['labels'] = labels[i].argmax(dim=1)
            targets.append(d)

        _, feats_list = self.backbone(video, return_intermediates=True)

        pooled_feats = []
        for conv3d, feat5d in zip(self.temporal_convs, feats_list):
            # [B, C, T, H, W]
            x = feat5d.permute(0, 4, 1, 2, 3)
            x = F.relu(conv3d(x))  # [B, 256, T, H, W]
            x = x.mean(dim=2)  # [B, 256, H, W]
            pooled_feats.append(x)

        raw_feats = OrderedDict({
            name: pooled_feats[i]
            for i, name in enumerate(["p2", "p3", "p4", "p5"])
        })

        features = self.fpn(raw_feats)

        losses = {}
        proposals, rpn_losses = self.rpn_network(image_list, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        losses.update(rpn_losses)
        losses.update(detector_losses)
        return detections, losses


if __name__ == '__main__':
    batch_size = 4
    channels = 3
    time_frames = 16
    img_size = 224
    num_classes = 5
    video = torch.rand(batch_size, channels, time_frames, img_size, img_size)

    sample_boxes = []
    sample_labels = []

    for b in range(batch_size):

        num_boxes = 3
        boxes_tensor = torch.tensor([
            [50, 50, 100, 100],
            [70, 70, 150, 150],
            [30, 30, 80, 80]
        ], dtype=torch.float)

        box_list = BoxList(boxes_tensor, (img_size, img_size), mode="xyxy")
        sample_boxes.append(box_list)

        labels = torch.zeros(num_boxes, num_classes)
        for i in range(num_boxes):
            class_idx = torch.randint(0, num_classes, (1,)).item()
            labels[i, class_idx] = 1.0
        sample_labels.append(labels)

    model = HieraActionDetector(num_classes=num_classes, use_focal_loss=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    video = video.to(device)
    sample_boxes = [b.to(device) for b in sample_boxes]
    sample_labels = [l.to(device) for l in sample_labels]
    model.eval()
    with torch.no_grad():
        outputs, losses = model(video, sample_boxes, sample_labels)
    print(losses)
    print(f"{type(outputs)}")
    if isinstance(outputs, list):
        print(f"detections: {len(outputs)}")
        det = outputs[0]
        print(f"detection boxes {det['boxes'].shape}")
        print(f"detection scores {det['scores'].shape}")
        print(f"detection labels {det['labels'].shape}")
        print(det)
    else:
        print(f"{outputs.shape}")
