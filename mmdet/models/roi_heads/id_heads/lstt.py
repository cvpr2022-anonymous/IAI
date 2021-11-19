import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmcv.runner import ModuleList

from mmdet.models.builder import HEADS

from mmcv.runner import BaseModule, auto_fp16, force_fp32
from networks.layers.transformer import LongShortTermTransformer
from networks.layers.position import PositionEmbeddingSine
from networks.layers.basic import ConvGN, seq_to_2d


def generate_permute_matrix(dim, num, keep_first=True):
    all_matrix = []
    for idx in range(num):
        random_matrix = torch.eye(dim).cuda()
        if keep_first:
            fg = random_matrix[1:][torch.randperm(dim - 1)]
            random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        else:
            random_matrix = random_matrix[torch.randperm(dim)]
        all_matrix.append(random_matrix)
    return torch.stack(all_matrix, dim=0)

@HEADS.register_module()
class LSTTBlock(BaseModule):

    def __init__(self,
                 max_obj_num=50,
                 lstt_num=3,
                 in_channels=256,
                 feat_channels=256,
                 self_heads=3,
                 attn_heads=2):
        super().__init__()
        self.max_obj_num = max_obj_num

        self.long_term_mem_gap = 5 #9999

        self.LSTT = LongShortTermTransformer(lstt_num,
                                             feat_channels,
                                             self_heads,
                                             attn_heads,
                                             return_intermediate=True)

        self.patch_wise_id_bank = nn.Conv2d(
            max_obj_num + 1, feat_channels, kernel_size=33, stride=32, padding=16)

        self.pos_generator = PositionEmbeddingSine(
            feat_channels // 2, normalize=True)

        self._init_weight()

        self.restart()
        self.id_emb_time = 0
        self.lstt_forward_time = 0
        self.iter = 0

    def _init_weight(self):
        nn.init.xavier_uniform_(self.patch_wise_id_bank.weight)

    def restart(self, batch_size=1, enable_id_shuffle=True):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None
        self.curr_obj_nums = 0

        self.long_term_memories = None
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        self.use_pred = False

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size)
        else:
            self.id_shuffle_matrix = None

        self.first_inst_exist = False

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        return id_emb

    def assign_identity(self, one_hot_mask):
        import time

        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum(
                'bohw,bot->bthw', one_hot_mask, self.id_shuffle_matrix)

        try:
            id_emb = self.get_id_emb(one_hot_mask).view(
                self.batch_size, -1, self.enc_hw).permute(2, 0, 1)
        except:
            print('fuck')
            import pdb
            pdb.set_trace()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def update_size(self, enc_size):
        #self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]

    def LSTT_forward(self, curr_embs, long_term_memories, short_term_memories, long_term_id=None, pos_emb=None, size_2d=(14, 14)):

        n, c, h, w = curr_embs[-1].size()
        curr_emb = curr_embs[-1].view(n, c, h * w).permute(2, 0, 1)
        lstt_embs, lstt_memories = self.LSTT(curr_emb, long_term_memories, short_term_memories, long_term_id, pos_emb, size_2d)
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(*lstt_memories)

        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories

    def process_id(self, gt_ids):
        import copy
        ori_gt_ids = copy.deepcopy(gt_ids)
        for img_id in range(self.batch_size):
            ids_set = set()
            for frame in range(5):
                gt_id_set = set(gt_ids[img_id + frame * self.batch_size].cpu().numpy().tolist())
                new_ids = gt_id_set - ids_set
                if len(new_ids) > 0:
                    for new_id in new_ids:
                        for index, gt_id in enumerate(gt_id_set):
                            if new_id == gt_id:
                                gt_ids[img_id + frame * self.batch_size][index] = self.max_obj_num
                ids_set = ids_set | gt_id_set

        return ori_gt_ids

    def match_propogate_one_frame(self, enc_embs):
        self.frame_step += 1
        curr_enc_embs = enc_embs

        curr_lstt_output = self.LSTT_forward(
            curr_enc_embs, self.long_term_memories, self.short_term_memories, None, pos_emb=self.pos_emb, size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = curr_lstt_output

        self.lstt_curr_memories = lstt_curr_memories

        return lstt_embs

    def add_reference_frame(self, enc_embs=None, mask=None, obj_nums=None):
        frame_step = self.frame_step
        self.last_mem_step = frame_step

        curr_enc_embs, curr_one_hot_mask = enc_embs, mask

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for reference frame!')
            exit()

        if self.enc_size_2d is None:
            self.update_size(curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        self.pos_emb = self.get_pos_emb(curr_enc_embs[-1]).expand(self.batch_size, -1, -1, -1).view(
            self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb


        curr_lstt_output = self.LSTT_forward(
            curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d)


        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = curr_lstt_output

        self.lstt_curr_memories = lstt_curr_memories

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)

        self.short_term_memories = lstt_short_memories

        return lstt_embs

    def reset_memory(self, mask=None):

        self.frame_step = 1
        curr_enc_embs = self.curr_enc_embs
        curr_id_emb = self.assign_identity(mask)
        curr_lstt_output = self.LSTT_forward(
            curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = curr_lstt_output

        self.lstt_curr_memories = lstt_curr_memories
        self.long_term_memories = lstt_long_memories
        self.short_term_memories = lstt_short_memories

    def update_long_term_memory(self, new_long_term_memories):
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory, last_long_term_memory):
                updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def update_short_term_memory(self, curr_mask, new_inst_exist=False):
        curr_id_emb = self.assign_identity(curr_mask)

        lstt_curr_memories = self.lstt_curr_memories
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_v = lstt_curr_memories[layer_idx][1]
            curr_v = self.LSTT.layers[layer_idx].linear_V(
                curr_v + curr_id_emb)
            lstt_curr_memories[layer_idx][1] = curr_v
            lstt_curr_memories_2d.append([seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_size_2d),
                                          seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_size_2d)])

        self.short_term_memories = lstt_curr_memories_2d

        if (self.frame_step - self.last_mem_step >= self.long_term_mem_gap) or (new_inst_exist):
            self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step

    def forward(self, enc_embs, id_mask=None, new_inst_exist=False):
        """Forward function of LSTT block.

        Args:
            enc_feat (Tensor): backbone encoder feature of images.
            id_mask (Tensor): idenfication mask of multiple instances in images
                shape (batch_size, max_obj_num+1, ori_height, ori_width)

          Returns:

            - lstt output (Tensor): lstt output including
            lstt encode embedding, lstt short memory, lstt long memory
        """

        if self.enc_size_2d is None:
            self.update_size(enc_embs[-1].size()[2:])

        if self.frame_step == 0:
            curr_lstt_embs = self.add_reference_frame(enc_embs, id_mask)

            self.frame_step += 1
        else:
            curr_lstt_embs = self.match_propogate_one_frame(enc_embs)

        return curr_lstt_embs



