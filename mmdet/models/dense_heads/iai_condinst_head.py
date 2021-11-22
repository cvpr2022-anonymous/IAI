import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, kaiming_init
from mmcv.runner import force_fp32
from mmcv.ops.nms import batched_nms

from mmdet.core import (distance2bbox, multi_apply, bbox_overlaps,
                        reduce_mean, unmap, bbox2result_with_id)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from ..losses import cross_entropy, accuracy
import pycocotools.mask as mask_util

INF = 1e8
EPS = 1e-12
TEST_TIME = False

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def multiclass_nms(multi_bboxes,
                   multi_cls_scores,
                   multi_id_scores,
                   multi_kernels,
                   multi_points,
                   multi_strides,
                   cls_score_thr,
                   id_score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   is_first=False):
    bboxes = multi_bboxes.reshape(-1, 4)
    cls_scores = multi_cls_scores.max(dim=1)[0].reshape(-1)
    id_scores = multi_id_scores.max(dim=1)[0].reshape(-1)
    kernels = multi_kernels.reshape(-1, 169)
    points = multi_points.reshape(-1, 2)
    strides = multi_strides.reshape(-1, 1)

    # remove low scoring boxes
    valid_mask = (cls_scores > cls_score_thr) & (id_scores > id_score_thr)
    if is_first:
        scores = cls_scores + 0.5 * id_scores
    else:
        scores = 0.5 * cls_scores + id_scores
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    cls_inds = (cls_scores > cls_score_thr).nonzero(as_tuple=False).squeeze(1)
    id_inds = (id_scores > id_score_thr).nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, kernels, points, strides = \
        bboxes[inds], scores[inds], kernels[inds], points[inds], strides[inds]
    return_inds = inds
    if inds.numel() == 0:
       return bboxes, kernels, points, strides, return_inds

    dets, keep = batched_nms(bboxes, scores, torch.ones(scores.shape), nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, kernels[keep], points[keep], strides[keep], return_inds[keep]

def dice_coefficient(x, target):
    eps = 1e-5
    n_instance = x.size(0)
    x = x.reshape(n_instance, -1)
    target = target.reshape(n_instance, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if (l != num_layers - 1):
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1 (out_channels = 1)
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0,
        w * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shifts_y = torch.arange(0,
        h * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]

@HEADS.register_module()
class IAICondInstHead(AnchorFreeHead):
    """IAICondInstHead
       add new id head to original CondInst head
    """
    def __init__(self,
                 num_classes,
                 max_obj_num,
                 in_channels,
                 stacked_convs=4,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_id=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 **kwargs):
        self.max_obj_num = max_obj_num
        self.id_out_channels = self.max_obj_num
        super(IAICondInstHead, self).__init__(num_classes, in_channels, **kwargs)
        self.strides = strides
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_on_bbox = norm_on_bbox

        # conv
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        # add new loss id
        if loss_id is not None:
            self.loss_id = build_loss(loss_id)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        # new id head consists of 1x1 convolution layers
        self.id_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            if i <= 0:
                self.id_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.conv_id = nn.Conv2d(
            self.feat_channels,
            self.id_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels,
            4,
            3,
            padding=1)
        self.conv_centerness = nn.Conv2d(
            self.feat_channels,
            1,
            3,
            padding=1)
        self.controller = nn.Conv2d(
            self.feat_channels,
            169,
            3,
            padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        # mask branch
        self.mask_refine = nn.ModuleList()
        in_features = ['p3', 'p4', 'p5']
        for in_feature in in_features:
            conv_block = []
            conv_block.append(
                nn.Conv2d(self.feat_channels,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())
            conv_block = nn.Sequential(*conv_block)
            self.mask_refine.append(conv_block)
        # mask head
        tower = []
        for i in range(self.stacked_convs):
            conv_block = []
            conv_block.append(
                nn.Conv2d(128,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())

            conv_block = nn.Sequential(*conv_block)
            tower.append(conv_block)

        tower.append(
            nn.Conv2d(128,
                      8,
                      kernel_size=1,
                      stride=1))
        self.mask_head = nn.Sequential(*tower)

        # conditional convs
        self.weight_nums = [80, 64, 8]
        self.bias_nums = [8, 8, 1]
        self.mask_out_stride = 4

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.id_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_id, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        kaiming_init(self.mask_refine)
        kaiming_init(self.mask_head)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    4.
        """
        id_scores = []
        cls_scores = []
        bbox_preds = []
        centernesses = []
        kernel_preds = []
        for i, (x, scale) in enumerate(zip(feats, self.scales)):
            cls_feat = x
            id_feat = x
            reg_feat = x

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for id_conv in self.id_convs:
                id_feat = id_conv(id_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            cls_score = self.conv_cls(cls_feat)
            id_score = self.conv_id(id_feat)
            bbox_pred = scale(self.conv_reg(reg_feat)).float()
            if self.norm_on_bbox:
                bbox_pred = F.relu(bbox_pred) * self.strides[i]
            else:
                bbox_pred = bbox_pred.exp()
            centerness = self.conv_centerness(reg_feat)
            kernel_pred = self.controller(reg_feat)

            # mask feat
            if i == 0:
                mask_feat = self.mask_refine[i](x)
            elif i <= 2:
                x_p = self.mask_refine[i](x)
                target_h, target_w = mask_feat.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                mask_feat = mask_feat + x_p

            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)
            id_scores.append(id_score)
            centernesses.append(centerness)
            kernel_preds.append(kernel_pred)

        mask_feat = self.mask_head(mask_feat)

        return cls_scores, id_scores, bbox_preds, centernesses, kernel_preds, mask_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             id_scores,
             bbox_preds,
             centernesses,
             kernel_preds,
             mask_feats,
             gt_bboxes,
             gt_labels,
             gt_ids,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_list=None,
             is_first=False):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ids=None,
                      gt_ori_ids=None,
                      use_pred=False,
                      is_first=False,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        pass

        return None

    def mask_heads_forward(self, features, weights, biases, num_instances):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                         w,
                         bias=b,
                         stride=1,
                         padding=0,
                         groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def relative_coordinate_feature_generator(self, mask_feat, instance_locations, strides):
        # obtain relative coordinate features for mask generator
        num_instance = len(instance_locations)
        H, W = mask_feat.size()[1:]
        locations = compute_locations(H,
                                      W,
                                      stride=8,
                                      device=mask_feat.device)
        relative_coordinates = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1) * 8.0)
        relative_coordinates = relative_coordinates.to(dtype=mask_feat.dtype)
        coordinates_feat = torch.cat([
            relative_coordinates.view(num_instance, 2, H, W),
            mask_feat.repeat(num_instance, 1, 1, 1)], dim=1)
        coordinates_feat = coordinates_feat.view(1, -1, H, W)
        return coordinates_feat

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   id_scores,
                   bbox_preds,
                   centernesses,
                   kernel_preds,
                   mask_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   use_pred=False,
                   gt_bboxes=None,
                   gt_ori_ids=None,
                   gt_masks_list=None,
                   test_mode=False,
                   is_first=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        if is_first:
            self.curr_inst_id = [0 for i in range(len(img_metas))]
        mlvl_points, mlvl_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype,
            bbox_preds[0].device)

        mask_results_list = []
        if use_pred:
            max_h, max_w = 0, 0
            for img_id in range(len(img_metas)):
                pad_shape = img_metas[img_id]['pad_shape']
                h, w = pad_shape[:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)
            id_masks = cls_scores[0].new_zeros(len(img_metas), self.max_obj_num+1, max_h, max_w)
            id_masks[:, self.max_obj_num] = 1
            new_inst_exists = False

        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            id_score_list = [
                id_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            kernel_pred_list = [
                kernel_preds[i][img_id].detach() for i in range(num_levels)
            ]
            mask_feats_i = mask_feats[img_id]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            det_bboxes, det_id_scores, det_cls_scores, ori_masks, det_masks = self._get_bboxes_single(
                cls_score_list,
                id_score_list,
                bbox_pred_list,
                centerness_pred_list,
                kernel_pred_list,
                mask_feats_i,
                mlvl_points,
                mlvl_strides,
                img_shape,
                scale_factor,
                ori_shape,
                cfg,
                rescale,
                with_nms,
                test_mode=test_mode,
                is_first=is_first)

            if use_pred:
                keep = []
                new_pad_masks = []
                bg_mask = id_masks.new_ones((max_h, max_w), dtype=torch.bool)
                if ori_masks is not None:
                    ori_bg_mask = id_masks.new_ones(ori_masks[0].shape, dtype=torch.bool)
                    new_ori_masks = []
                for i in range(len(det_bboxes)):
                    mask = det_masks[i].bool()
                    pad = (0, max_w - mask.shape[1], 0, max_h - mask.shape[0])
                    pad_mask = F.pad(mask, pad, value=0)
                    new_pad_mask = bg_mask & pad_mask
                    bg_mask[new_pad_mask] = 0
                    area = pad_mask.sum()
                    if (area == 0) or float(new_pad_mask.sum()) / float(area) < 0.1:
                        continue
                    keep.append(i)
                    new_pad_masks.append(new_pad_mask)

                    if ori_masks is not None:
                        new_ori_mask = ori_bg_mask & ori_masks[i].bool()
                        ori_bg_mask[new_ori_mask] = 0
                        new_ori_masks.append(new_ori_mask)

                if len(keep) > 0:
                    det_bboxes = det_bboxes[keep]
                    det_id_scores = det_id_scores[keep]
                    if ori_masks is not None:
                        ori_masks = ori_masks[keep]
                    det_cls_scores = det_cls_scores[keep]

                from scipy.optimize import linear_sum_assignment
                if is_first:
                    det_ids = det_id_scores.new_ones(det_bboxes.shape[0]) * (self.max_obj_num - 1)
                else:
                    if len(det_id_scores) > 0:
                        new_id_scores = det_id_scores[:, self.max_obj_num-1].repeat(len(det_id_scores)-1,1)
                        id_scores_matrix = -torch.cat((det_id_scores.transpose(0,1), new_id_scores)).transpose(0,1).cpu()
                        row_ind, col_ind = linear_sum_assignment(id_scores_matrix)
                        det_ids = col_ind
                    if (not test_mode) and (len(gt_bboxes[img_id][add]) > 0):
                        det_ids = np.concatenate((gt_ori_ids[img_id][add].flatten(0).cpu().numpy(), det_ids))

                cls_scores_dict = {}
                det_obj_ids = []
                curr_max_id = self.curr_inst_id[img_id]
                for i in range(len(keep)):
                    id_pred = det_ids[i].item()
                    new_pad_mask = new_pad_masks[i]

                    if (is_first) or (id_pred >= curr_max_id):
                        new_inst_exists = True
                        id_pred = self.curr_inst_id[img_id]
                        self.curr_inst_id[img_id] += 1
                        if self.curr_inst_id[img_id] > self.max_obj_num-2:
                            self.curr_inst_id[img_id] = self.max_obj_num-2
                    if id_pred in det_obj_ids:

                        id_pred = self.curr_inst_id[img_id]
                        self.curr_inst_id[img_id] += 1
                        if self.curr_inst_id[img_id] > self.max_obj_num-2:
                            self.curr_inst_id[img_id] = self.max_obj_num-2

                    det_obj_ids.append(id_pred)
                    cls_scores_dict[id_pred] = det_cls_scores[i]
                    id_masks[img_id][id_pred] = new_pad_mask #* det_cls_scores[i].max()
                    id_masks[img_id][self.max_obj_num][new_pad_mask] = 0

            if len(keep) == 0:
                bbox_results = {}
                mask_results_list.append({})
                continue

            bbox_results = bbox2result_with_id(det_bboxes, det_bboxes.new_ones(det_bboxes.size(0)), det_obj_ids)

            mask_results = {}
            for i in range(len(keep)):
                id_pred = det_obj_ids[i]
                mask = new_ori_masks[i].bool().cpu().numpy()
                mask_results[id_pred] = mask
            mask_results_list.append(mask_results)

        if use_pred:
            return bbox_results, mask_results_list, id_masks, new_inst_exists, cls_scores_dict

    def _get_bboxes_single(self,
                           cls_scores,
                           id_scores,
                           bbox_preds,
                           centernesses,
                           kernel_preds,
                           mask_feat,
                           mlvl_points,
                           mlvl_strides,
                           img_shape,
                           scale_factor,
                           ori_shape,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           test_mode=False,
                           is_first=False):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (1, H, W).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_id_scores = []
        mlvl_centerness = []
        mlvl_kernels_pred = []
        flatten_mlvl_points = []
        flatten_mlvl_strides = []
        for cls_score, id_score, bbox_pred, centerness, kernel_pred, points, strides in zip(
                cls_scores, id_scores, bbox_preds, centernesses, kernel_preds,  mlvl_points, mlvl_strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            id_scores_pred = id_score.permute(1, 2, 0).reshape(
                -1, self.id_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, 169)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds, :]
                strides = strides[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                id_scores_pred = id_scores_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                kernel_pred = kernel_pred[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_id_scores.append(id_scores_pred)
            mlvl_centerness.append(centerness)
            mlvl_kernels_pred.append(kernel_pred)
            flatten_mlvl_strides.append(strides)
            flatten_mlvl_points.append(points)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_kernels_pred = torch.cat(mlvl_kernels_pred)

        flatten_mlvl_points = torch.cat(flatten_mlvl_points)
        flatten_mlvl_strides = torch.cat(flatten_mlvl_strides)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_id_scores = torch.cat(mlvl_id_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        padding = mlvl_id_scores.new_zeros(mlvl_id_scores.shape[0], 1)
        mlvl_id_scores = torch.cat([mlvl_id_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        id_score_thr = cfg.id_score_thr
        cls_score_thr = cfg.cls_score_thr
        det_bboxes, det_kernels_pred, det_points, det_strides, det_inds = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_id_scores,
            mlvl_kernels_pred,
            flatten_mlvl_points,
            flatten_mlvl_strides,
            cls_score_thr,
            id_score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            is_first=is_first)

        det_id_scores = mlvl_id_scores[det_inds]
        det_cls_scores = mlvl_scores[det_inds] * mlvl_centerness[det_inds].unsqueeze(1)

        # generate masks
        masks = None
        ori_masks = None
        if det_bboxes.shape[0] > 0:
            mask_head_params = det_kernels_pred
            num_instance = len(det_points)
            mask_head_inputs = self.relative_coordinate_feature_generator(
                mask_feat,
                det_points,
                det_strides)
            weights, biases = parse_dynamic_params(
                mask_head_params,
                8,
                self.weight_nums,
                self.bias_nums)
            mask_logits = self.mask_heads_forward(
                mask_head_inputs,
                weights,
                biases,
                num_instance)
            mask_logits = mask_logits.reshape(-1, 1, mask_feat.size(1), mask_feat.size(2))
            mask_logits = aligned_bilinear(mask_logits, 2).sigmoid()

            pred_global_masks = aligned_bilinear(mask_logits, 4)
            pred_global_masks = pred_global_masks[:, :, :img_shape[0], :img_shape[1]]
            ori_masks = F.interpolate(
                pred_global_masks,
                size=(ori_shape[0], ori_shape[1]),
                mode='bilinear',
                align_corners=False).squeeze(1)
            ori_masks.gt_(0.5)
            masks = aligned_bilinear(mask_logits, 4).squeeze(1)
            masks.gt_(0.5)
        return det_bboxes, det_id_scores, det_cls_scores, ori_masks, masks

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_ids_list, gt_masks_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, ids_list, bbox_targets_list, gt_inds_list = multi_apply(
            #self.get_targets_single,
            self._get_targets_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            gt_ids_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        ids_list = [ids.split(num_points, 0) for ids in ids_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        return labels_list, ids_list, bbox_targets_list, gt_inds_list

    def _get_targets_single(self,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           gt_ids,
                           points,
                           regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        # TODO make compatible
        if num_gts == 0:
            #raise NotImplementedError
            return gt_labels.new_full((num_points,), self.num_classes), gt_labels.new_full((num_points,), self.max_obj_num),\
                   gt_bboxes.new_zeros((num_points, 4)), gt_labels.new_zeros(num_points)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius

            # use masks to determine center region
            _, h, w = gt_masks.size()
            yys = torch.arange(0, h, dtype=torch.float32, device=gt_masks.device)
            xxs = torch.arange(0, w, dtype=torch.float32, device=gt_masks.device)

            m00 = gt_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_masks * xxs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_masks * yys[:, None]).sum(dim=-1).sum(dim=-1)
            center_xs = m10 / m00
            center_ys = m01 / m00
            center_xs = center_xs[None].expand(num_points, num_gts)
            center_ys = center_ys[None].expand(num_points, num_gts)

            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        ids = gt_ids[min_area_inds]
        #ids[min_area == INF] = self.num_classes  # set as BG
        ids[min_area == INF] = self.max_obj_num
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pot_gt_inds = min_area_inds[labels < self.num_classes]

        return labels, ids, bbox_targets, pot_gt_inds

    def get_targets_single(self, gt_bboxes, gt_labels, gt_masks, gt_ids,
                            points, regress_ranges, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_zeros(num_points), gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels.new_zeros(num_points), gt_labels.new_zeros(num_points)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            if gt_masks is not None:
                _, h, w = gt_masks.size()

                ys_ = torch.arange(0, h, dtype=torch.float32, device=gt_masks.device)
                xs_ = torch.arange(0, w, dtype=torch.float32, device=gt_masks.device)

                m00 = gt_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
                m10 = (gt_masks * xs_).sum(dim=-1).sum(dim=-1)
                m01 = (gt_masks * ys_[:, None]).sum(dim=-1).sum(dim=-1)
                center_xs = m10 / m00
                center_ys = m01 / m00
                center_xs = center_xs[None].expand(num_points, num_gts)
                center_ys = center_ys[None].expand(num_points, num_gts)
            else:
                center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
                center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) \
            & (max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        ids = gt_ids[min_area_inds]
        ids[min_area == INF] = self.max_obj_num

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_ind = min_area_inds[labels < self.num_classes]

        return labels, ids, bbox_targets, gt_ind

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single(
                featmap_sizes[i],
                self.strides[i],
                dtype,
                device)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        strides = points[:,0] * 0 + stride
        return points, strides

    def simple_test(self, x, img_metas, rescale=False, is_first=False):
        outputs = self(x)
        bbox_inputs = outputs + (img_metas, self.test_cfg, rescale)

        bbox_results, segm_results, id_masks, new_inst_exists, cls_scores_dict = self.get_bboxes(*bbox_inputs, use_pred=True, test_mode=True, is_first=is_first)

        return [(bbox_results, segm_results)], id_masks, new_inst_exists, cls_scores_dict

