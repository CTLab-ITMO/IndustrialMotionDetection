import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from typing import List, Tuple
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from dataclasses import dataclass
from collections import OrderedDict
from models.image_list import ImageList
from models.pooler import make_3d_pooler


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        x = self.proj(x)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0)


@dataclass
class ROIPoolingCfg:
    POOLER_RESOLUTION: int = 7
    POOLER_SCALE: float = 0.0625
    POOLER_SAMPLING_RATIO: int = 0
    POOLER_TYPE: str = 'align3d'
    MEAN_BEFORE_POOLER: bool = True


class VisionTransformer(nn.Module):
    """
    Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=80,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches  # 8x14x14
        self.use_checkpoint = use_checkpoint
        self.grid_size = [img_size//patch_size, img_size//patch_size]  # [14,14]

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.fc_norm = None

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # --- Region Proposal Network Code:
        def _default_anchorgen():
            anchor_sizes = ((32, 64, 128, 256, 512),)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            return torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)

        out_channels = embed_dim

        rpn_anchor_generator = _default_anchorgen()
        rpn_head = torchvision.models.detection.rpn.RPNHead(
            out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0]
        )

        # number of proposals to keep after applying NMS.
        rpn_pre_nms_top_n_train=2000
        rpn_pre_nms_top_n_test=1000
        rpn_post_nms_top_n_train=2000
        rpn_post_nms_top_n_test=1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # NMS threshold used for postprocessing the RPN proposals
        rpn_nms_thresh=0.7

        # minimum IoU between the anchor and the GT box so that they can be
        # considered as positive during training of the RPN.
        rpn_fg_iou_thresh=0.7

        # maximum IoU between the anchor and the GT box so that they can be
        # considered as negative during training of the RPN.
        rpn_bg_iou_thresh=0.3

        # --- BalancedPositiveNegativeSampler ---

        # number of anchors that are sampled during training of the RPN for computing the loss
        rpn_batch_size_per_image=256

        # proportion of positive anchors in a mini-batch during training of the RPN
        rpn_positive_fraction=0.5
        # ---------------------------------------

        # only return proposals with an objectness score greater than score_thresh
        rpn_score_thresh=0.0

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
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2)

        representation_size = 1024
        resolution =  box_roi_pool.output_size[0]
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            out_channels * resolution**2,
            representation_size)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            representation_size,
            num_classes)

        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5

        box_batch_size_per_image=512
        box_positive_fraction=0.25

        bbox_reg_weights=None
        box_score_thresh=0.05
        box_nms_thresh=0.5
        box_detections_per_img=100

        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,

            # Faster R-CNN training
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,

            # Faster R-CNN inference
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        self.bbox_head = nn.Linear(embed_dim, num_classes * 4)

        # rois setting
        self.head_cfg = ROIPoolingCfg()
        self.pooler = make_3d_pooler(self.head_cfg)
        resolution = self.head_cfg.POOLER_RESOLUTION
        self.max_pooler = nn.MaxPool2d((resolution, resolution))
        # --- End of RPN Code implementation

        self.test_ext = (0.1, 0.05)
        self.proposal_per_clip = 300000

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, proposals, labels):
        # x shape - [b, c, t, h, w]
        # proposals - List[b, BoxList[n]]
        # labels - [b, n, num_classes]

        # The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
        # image, and should be in ``0-1`` range. Different images can have different sizes.
        images = x[:, :, x.shape[2] // 2] # video [b, c, h, w]

        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        # During training, the model expects both the input tensors and a targets (list of dictionary),
        # containing:
        # - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
        #   ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        # - labels (``Int64Tensor[N]``): the class label for each ground-truth box

        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = proposals[i].bbox  # [N, 4] float tensor
            d['labels'] = labels[i].argmax(dim=1) # Tensor[N]
            targets.append(d)

        x = self.patch_embed(x)
        B, width, t, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.reshape(t, -1, width)
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [h, w], 0).reshape(1, -1, width)
            x = x + pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)  # [b thw=8x14x14 c=768]
        x = x.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)  # [b c t h w]
        x = x.mean(dim=2, keepdim=False)  # [b c h w]

        losses = {}
        features = OrderedDict([("0", x)])
        proposals, rpn_losses = self.rpn_network(image_list, features, targets)

        # proposals = BoxList(proposals[0], (224, 224))
        # proposals - List[b, Tensor[N, 4]]
        detections, detector_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)

        losses.update(rpn_losses)
        losses.update(detector_losses)

        # rois = self.pooler(x, proposals)  # [n c 7 7]
        # rois = self.max_pooler(rois).view(rois.size(0), -1)  # [n c]
        # return rois, rpn_losses
        return detections, losses

    def sample_box(self, boxes):
        proposals = []
        num_proposals = self.proposal_per_clip
        for boxes_per_image in boxes:
            num_boxes = len(boxes_per_image)

            if num_boxes > num_proposals:
                choice_inds = torch.randperm(num_boxes)[:num_proposals]
                proposals_per_image = boxes_per_image[choice_inds]
            else:
                proposals_per_image = boxes_per_image

            proposals_per_image = proposals_per_image.random_aug(0.2, 0.1, 0.1, 0.05)
            proposals.append(proposals_per_image)
        return proposals

    def forward(self, x, boxes, labels):
        if self.training:
            proposals = self.sample_box(boxes)
        else:
            proposals = [box.extend(self.test_ext) for box in boxes]
        # --- Region Proposal Network Code:
        # rois, rpn_losses = self.forward_features(x, proposals, labels)
        detections, losses = self.forward_features(x, proposals, labels)
        # cls = self.head(rois)
        # bboxes = self.bbox_head(rois)
        # return cls, bboxes, rpn_losses
        return detections, losses
        # ---


def interpolate_pos_embed_online(
    pos_embed, orig_size: Tuple[int], new_size: Tuple[int], num_extra_tokens: int
):
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    embedding_size = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(
        -1, orig_size[0], orig_size[1], embedding_size
    ).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_size, mode="bicubic", align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    return model
