import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from datasets.solo_dataset import BuildDataset, BuildDataLoader
from functools import partial
import cv2
import copy
import numpy as np

import matplotlib.pyplot as plt


class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 device,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.33,
                                      ins_thresh=0.6,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.6)):
        """
        Args:
            num_classes: number of categories excluding the background category
            device: device to run on
            in_channels: number of channels in the input feature map
            seg_feat_channels: number of channels in the feature map for segmentation
            stacked_convs: number of stacked convolutional layers
            strides: the strides for feature map
            scale_ranges: the range of scales for each feature map
            epsilon: the value to be added to the denominator for numerical stability
            num_grids: the number of grids for each feature map
            cate_down_pos: the downsample position for category branch
            with_deform: whether to use deformable convolution
            mask_loss_cfg: the configuration for mask loss
            cate_loss_cfg: the configuration for category loss
            postprocess_cfg: the configuration for postprocessing
        """
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        self.device = device

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
        
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            ),
        ])

        self.cate_out = nn.ModuleList([nn.Conv2d(in_channels=self.in_channels, out_channels=self.cate_out_channels, kernel_size=3, stride=1, padding=1, bias=True) ,
                                        nn.Sigmoid()])

        self.ins_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels+2, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.ReLU()
            )
        ])
        self.conv_layers = nn.ModuleList([nn.Sequential(
                                            nn.Conv2d(in_channels=self.in_channels, 
                                                    out_channels=self.in_channels, 
                                                    kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.GroupNorm(num_groups=32, num_channels=256),
                                            nn.ReLU()
                                        ) for _ in range(6)  # Creating 6 identical sequential layers
                                    ])
        self.ins_head.extend(self.conv_layers)
        self.ins_out_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.seg_num_grids[i]**2, 
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        ) for i in range(5) ])
        

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the weights
        # Initialize weights for each layer in cate_head
        for m in self.cate_head:
            if isinstance(m[0], nn.Conv2d):
                nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='relu')
                if m[0].bias is not None:
                    nn.init.constant_(m[0].bias, 0)
            if isinstance(m[1], nn.GroupNorm):
                nn.init.constant_(m[1].weight, 1)
                nn.init.constant_(m[1].bias, 0)

        # Initialize weights for cate_out layer
        for m in self.cate_out:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize weights for ins_head layers
        if isinstance(self.ins_head, nn.Conv2d):
            nn.init.kaiming_normal_(self.ins_head.weight, mode='fan_out', nonlinearity='relu')
            if self.ins_head.bias is not None:
                nn.init.constant_(self.ins_head.bias, 0)

        # Initialize weights for the conv layers in ins_head
        for m in self.conv_layers:
            if isinstance(m[0], nn.Conv2d):
                nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='relu')
                if m[0].bias is not None:
                    nn.init.constant_(m[0].bias, 0)
            if isinstance(m[1], nn.GroupNorm):
                nn.init.constant_(m[1].weight, 1)
                nn.init.constant_(m[1].bias, 0)

        # Initialize weights for the output layers in ins_out_list
        for m in self.ins_out_list:
            if isinstance(m[0], nn.Conv2d):
                nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='sigmoid')
                if m[0].bias is not None:
                    nn.init.constant_(m[0].bias, 0)


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level, new_fpn_list,list(range(len(new_fpn_list))),eval=eval)
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):

        new_fpn_list = []

        for i, feat in enumerate(fpn_feat_list):
            H_f = int(800 / self.strides[i])
            W_f = int(1088/ self.strides[i])
            if i == 0 or i == len(fpn_feat_list) -1: 
                images_per_feat = []
                for num in range(feat.shape[0]):
                    images_per_feat.append(F.interpolate(feat[num].unsqueeze(0), size=(H_f, W_f), mode='bilinear'))
                new_fpn_list.append(torch.vstack(images_per_feat))
            else:
                new_fpn_list.append(feat)
        return new_fpn_list


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        num_grid = self.seg_num_grids[idx]  # current level grid
        cat_pred_batch = []
        mask_pred_batch = []
        for num in range(fpn_feat.shape[0]):

            cate_feat = F.interpolate(fpn_feat[num].unsqueeze(0), size=(num_grid , num_grid ) , mode = 'bilinear')
            # Category branch processing
            for cate_layer in self.cate_head:
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.cate_out[0](cate_feat)
            cate_pred = self.cate_out[1](cate_pred)

            cat_pred_batch.append(cate_pred)

            

            mask_feat = fpn_feat[num].clone().unsqueeze(0).to(self.device)
            mask_feat = F.interpolate(mask_feat, size=(mask_feat.shape[2]*2 , mask_feat.shape[3]*2 ), mode = 'bilinear').to(self.device)
            x_channel, y_channel = torch.meshgrid(torch.arange(mask_feat.shape[2], device=self.device), torch.arange(mask_feat.shape[3], device=self.device))
            x_channel = x_channel.to(self.device)
            y_channel = y_channel.to(self.device)

            # Normalize between -1 and 1
            x_channel_norm = x_channel - torch.min(x_channel)
            x_channel_norm = x_channel_norm/torch.max(x_channel_norm)
            x_channel_norm = (x_channel_norm - 0.5)/0.5 

            y_channel_norm = y_channel - torch.min(y_channel)
            y_channel_norm = y_channel_norm/torch.max(y_channel_norm)
            y_channel_norm = (y_channel_norm - 0.5)/0.5 

            mask_feat = torch.cat([mask_feat.squeeze(0), x_channel_norm.unsqueeze(0), y_channel_norm.unsqueeze(0)], dim=0).unsqueeze(0)
            
            for mask_layer in self.ins_head:
                mask_feat = mask_layer(mask_feat)

            mask_pred = self.ins_out_list[idx](mask_feat)

            mask_pred_batch.append(mask_pred)

        cate_pred = torch.vstack(cat_pred_batch)
        ins_pred = torch.vstack(mask_pred_batch)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            image_h = int(800 / 4)
            image_w = int(1088 / 4)
            ins_pred = F.interpolate(ins_pred, size=(image_h, image_w), mode='bilinear')

            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        lambda_cat = self.cate_loss_cfg['weight']
        lambda_msk = self.mask_loss_cfg['weight']
        
        cat_loss = lambda_cat * self.FocalLoss(cate_gts_list, cate_pred_list)
        msk_loss = lambda_msk * self.MaskLoss(ins_gts_list, ins_pred_list, ins_ind_gts_list)
        final_loss = cat_loss + msk_loss

        # print("c loss : ", cat_loss)
        # print("m loss : ", msk_loss)
        
        return cat_loss, msk_loss, final_loss
    
    def MaskLoss(self, msk_tar, msk_pred, act_msk):
      N_pos = 0
      tot_loss = 0
      loss_each_image = []
      for i in range(len(msk_tar)):                  # one image
        loss_every_level = []
        one_loss = 0
        for each_level in range(5):
          flattened_act_level = act_msk[i][each_level]               # Shape : (S^2,)
          N_pos += torch.sum(flattened_act_level)
          idx = torch.where(flattened_act_level!=0)[0]
          if len(idx) == 0:
            continue
          active_msk_tar_level = msk_tar[i][each_level][idx]      
          active_msk_pred_level = msk_pred[each_level][i][idx]
          for k in range(active_msk_pred_level.shape[0]):
            one_loss += self.DiceLoss(active_msk_pred_level[k], active_msk_tar_level[k])
          loss_every_level.append(one_loss)
        loss_each_image.append(sum(loss_every_level)/5)
      tot_loss = sum(loss_each_image)/len(loss_each_image)
      return tot_loss/N_pos


    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        mask_pred = mask_pred.to(self.device)
        mask_gt = mask_gt.to(self.device)
        dice_sum = torch.sum(2*mask_pred * mask_gt) / (torch.sum(torch.pow(mask_pred, 2)) + torch.sum(torch.pow(mask_gt, 2) + 0.000001))
        return 1 - dice_sum

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cat_tar, cat_pred):
        ## TODO: compute focalloss
        cat_tar_flattened = torch.cat([torch.cat([each_level.flatten() for each_level in each_image]) for each_image in cat_tar]).type(torch.long)   # Shape : batch_size x (1600 + 1296 + 576 + 256 + 144) (3872)
        one_hot_cat_tar = F.one_hot(cat_tar_flattened, num_classes=4)[:,1:]
        one_hot_cat_tar_flattened = one_hot_cat_tar.clone().flatten().to(self.device)                                                                   # Shape : len x 3 flattened

        cat_pred_flattened = torch.cat([each_channel.permute(0,2,3,1).reshape(-1,3) for each_channel in cat_pred])                                   # Shape : batch_size x (1600 + 1296 + 576 + 256 + 144) flattened
        new_cat = cat_pred_flattened.clone().flatten().to(self.device)

        idx = torch.where(one_hot_cat_tar_flattened!=0)[0]

        alphas = torch.ones_like(one_hot_cat_tar_flattened) * self.cate_loss_cfg["alpha"]

        alphas[idx] = 1 - self.cate_loss_cfg["alpha"]
        gamma = self.cate_loss_cfg["gamma"]

        ones = torch.ones(idx.shape[0])
        new_cat[idx] = 1 - new_cat[idx]
        focal_loss = torch.sum(-alphas * torch.pow(1 - new_cat, gamma) * torch.log(new_cat + 0.0000001)) / (3*cat_tar_flattened.shape[0])

        return focal_loss

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training

        featmap_sizes = []
        for i,pred in enumerate(ins_pred_list):
          featmap_sizes.append([ins_pred_list[i].shape[2], ins_pred_list[i].shape[3]] )
        
        featmap_sizes = [featmap_sizes for i in range(len(mask_list))]

        # self.target_single_img(bbox_list[0], label_list[0], mask_list[0], featmap_sizes)
        
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.target_single_img, 
                                                                              bbox_list,
                                                                              label_list, 
                                                                              mask_list, 
                                                                              featmap_sizes)


        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        # initial the output list, each entry for one featmap
        gt_bboxes_raw = gt_bboxes_raw.cpu()
        gt_labels_raw = gt_labels_raw.cpu()
        gt_masks_raw = gt_masks_raw.cpu()
        h, w = gt_masks_raw.shape[1], gt_masks_raw.shape[2]
        # Area normalized by height and width of image
        area   = torch.sqrt((gt_bboxes_raw[:,2] - gt_bboxes_raw[:,0]) * (gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1]))
        region = torch.zeros((gt_masks_raw.shape[0],4)).to(self.device)

        for i in range(gt_masks_raw.shape[0]):
            centre_of_mass = ndimage.measurements.center_of_mass(gt_masks_raw[i,:,:].cpu().numpy()) 
            # region[i, 0] = centre_of_mass[0]
            # region[i, 1] = centre_of_mass[1]
            region[i, 0] = centre_of_mass[1]
            region[i, 1] = centre_of_mass[0]

        region[:,2] = (gt_bboxes_raw[:,2] - gt_bboxes_raw[:,0]) *0.2/w          #Width 
        region[:,3] = (gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1]) *0.2/h          #Height
        
        #Rescaling Centers
        region[:,0] = region[:,0]/w 
        region[:,1] = region[:,1]/h
        
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []
        # no_of_zeros = 0 
        for i, size in enumerate(featmap_sizes):
            if (i==0):
                idx = area<96
            elif i==1:
                idx = torch.logical_and(area<192,area>48)
            elif i==2: 
                idx = torch.logical_and(area<384,area>96)
            elif i==3: 
                idx = torch.logical_and(area<768,area>192)
            elif i==4: 
                idx = area>=384
            
            idx = idx.cpu()
            if torch.sum(idx) == 0:
                cat_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i]))
                ins_label = torch.zeros((self.seg_num_grids[i]**2, size[0], size[1]))
                ins_index_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool)

                cate_label_list.append(cat_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_index_label)
                # no_of_zeros += 1
                continue
            
            region_idx = region[idx,:]
            left_ind   = ((region_idx[:,0] - region_idx[:,2]/2)*self.seg_num_grids[i]).int()
            right_ind  = ((region_idx[:,0] + region_idx[:,2]/2)*self.seg_num_grids[i]).int()
            top_ind    = ((region_idx[:,1] - region_idx[:,3]/2)*self.seg_num_grids[i]).int()
            bottom_ind = ((region_idx[:,1] + region_idx[:,3]/2)*self.seg_num_grids[i]).int()

            left   = torch.max(torch.zeros_like(left_ind)                            , left_ind  )
            right  = torch.min(torch.ones_like(right_ind)*(self.seg_num_grids[i] - 1)  , right_ind )
            top    = torch.max(torch.zeros_like(top_ind)                            , top_ind   )
            bottom = torch.min(torch.ones_like(bottom_ind)*(self.seg_num_grids[i] - 1) , bottom_ind)

            xA = torch.max(left    , (region_idx[:,0]*self.seg_num_grids[i]).int() - 1)
            xB = torch.min(right   , (region_idx[:,0]*self.seg_num_grids[i]).int() + 1)
            yA = torch.max(top     , (region_idx[:,1]*self.seg_num_grids[i]).int() - 1)
            yB = torch.min(bottom  , (region_idx[:,1]*self.seg_num_grids[i]).int() + 1)

            #Size of ins_label = S^2 x 2H_feat x 2W_feat
            cat_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i]))

            ins_label = torch.zeros((self.seg_num_grids[i]**2, size[0], size[1]))

            # ins_label = torch.zeros((self.seg_num_grids[i]**2, h, w))
            
            ins_index_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool)

            mask_interpolate = torch.nn.functional.interpolate(gt_masks_raw[idx,:,:].unsqueeze(0),
                                                               (size[0],size[1]),mode='nearest').squeeze(0)
            mask_interpolate[mask_interpolate > 0.5] = 1
            mask_interpolate[mask_interpolate < 0.5] = 0

            for j in range(xA.size(0)):

              cat_label[yA[j]:yB[j]+1 , xA[j]:xB[j]+1] = gt_labels_raw[idx][j]
              
              flag_matrix = torch.zeros(cat_label.shape)

              flag_matrix[yA[j]:yB[j]+1 , xA[j]:xB[j]+1] = 1
              positive_index = (torch.flatten(flag_matrix) > 0)

              ins_label[positive_index,:,:] = mask_interpolate[j,:,:]
              # ins_label[positive_index,:,:] = gt_masks_raw[idx,:,:,:][j,0,:,:]
              ins_index_label = torch.logical_or(ins_index_label,positive_index)

            
            # if ins_index_label.sum() == 0:
            #   print(self.seg_num_grids[i])
            #   print(region_idx[:,1])
            #   print(region_idx[:,3])
            #   # print(((region_idx[:,1] - region_idx[:,3]/2)*self.seg_num_grids[i]).int())
            #   print(xA,xB,yA,yB)
            #   print((flag_matrix >0).sum())
            #   print(flag_matrix[flag_matrix>0])
            #   while(1):
            #     pass

            cate_label_list.append(cat_label)
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_index_label)
        


        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list):

        ## TODO: finish PostProcess

        """
        Post-process the output of the model to obtain the final masks and class labels.

        Args:
            ins_pred_list (list of tensor): Instance masks for each category, shape (S^2, image_h/4, image_w/4).
            cate_pred_list (list of tensor): Category scores for each level, shape (S^2, C-1).

        Returns:
            final_msks (tensor): The final masks, shape (2, image_h, image_w).
            final_classes (tensor): The final class labels, shape (2,).
        """
        flattened_cats = torch.cat([torch.flatten(cat_each_level, start_dim=0, end_dim=1) for cat_each_level in cate_pred_list])   # Shape (5*S^2, C-1)
        flattened_msks = torch.cat(ins_pred_list)    # Shape (5*S^2, image_h/4, image_w/4)

        c_max, c_max_ind = torch.max(flattened_cats, dim=1)

        idx = torch.where(c_max > self.postprocess_cfg["cate_thresh"])[0]
        above_thresh_cmax = c_max[idx]
        above_thresh_cmax_ind = c_max_ind[idx]

        above_thres_msks = flattened_msks[idx]

        scores = torch.stack([torch.mean(above_thres_one_msk[torch.where(above_thres_one_msk > self.postprocess_cfg["ins_thresh"])]*above_thresh_cmax_each) 
                                        for above_thres_one_msk, above_thresh_cmax_each in zip(above_thres_msks.float(), above_thresh_cmax.float())])
        scores = torch.nan_to_num(scores, nan=0.)

        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_msks = above_thres_msks[sorted_indices]
        sorted_classes = above_thresh_cmax_ind[sorted_indices]

        nms_scores = self.MatrixNMS(sorted_msks, sorted_scores)
        sorted_nms_scores, sorted_nms_ind = torch.sort(nms_scores, descending=True)
        sorted_masks_final = sorted_msks[sorted_nms_ind]

        sorted_nms_classes = sorted_classes[sorted_nms_ind]

        first_five_msks = sorted_masks_final[:2]
        final_classes = sorted_nms_classes[:2]

        final_msks = torch.nn.functional.interpolate(first_five_msks.unsqueeze(0), size=(800, 1088), mode="bilinear")

        return final_msks, final_classes


    # This function perform matrix NMS
    # Input:
        # sorted_masks: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        n = len(sorted_scores)
        sorted_masks = sorted_masks.reshape(n, -1)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)

        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        return sorted_scores * decay

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        fig = plt.figure(figsize=(20, 20))
        for j in range(len(ins_gts_list)):
            for i in range(len(ins_gts_list[j])):
                plt.subplot(1,5,i+1)
                plt.title('FPN level: ' + str(i+1))
                plt.axis('off')
                no_mask = img[j,:,:,:].cpu().numpy()
                no_mask = np.clip(no_mask, 0, 1)
                no_mask = no_mask.transpose(1,2,0)
                plt.imshow(no_mask)
                if sum(ins_ind_gts_list[j][i]) == 0 :
                    continue

                index = ins_ind_gts_list[j][i] > 0
                label = torch.flatten(cate_gts_list[j][i])[index]
                mask = ins_gts_list[j][i][index,:,:]
                mask = torch.unsqueeze(mask,1)

                reshaped_mask = F.interpolate(mask,(img.shape[2],img.shape[3]),mode='bilinear')
                
                combined_mask = np.zeros((img.shape[2],img.shape[3],img.shape[1]))

                for idx, l in enumerate(label):
                    if l == 3:
                        combined_mask[:,:,0] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
                    if l == 2:
                        combined_mask[:,:,1] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
                    if l == 1:
                        combined_mask[:,:,2] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
                
                origin_img = img[j,:,:,:].cpu().numpy()
                origin_img = np.clip(origin_img, 0, 1)
                origin_img = origin_img.transpose(1,2,0)
                index_to_mask = np.where(combined_mask > 0)
                masked_image = copy.deepcopy(origin_img)
                masked_image[index_to_mask[0],index_to_mask[1],:] = 0

                mask_to_plot = (combined_mask + masked_image)
                mask_to_plot = np.clip(mask_to_plot, 0, 1)
                plt.imshow(mask_to_plot)
                
                
                
        plt.show()

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

    def visualize_nms_image(self,img_orig, ff_msks, classes,thresh=0.2, transp=0.2):
        """
        Visualize the inference segmentation in img
        Input:
            img_orig: (3, H, W)
            ff_msks: (1, H, W)
            classes: list, (n_class,)
            thresh: float, 0.2
            transp: float, 0.2
        Output:
            im: (H, W, 3)
        """
        img_orig = img_orig.cpu()
        ff_msks = ff_msks.cpu()
        manipulated_mask = torch.where(ff_msks>thresh, 1, 0)
        edit_im = img_orig.detach().cpu().numpy().copy()

        for x in range(len(manipulated_mask[0])):
            negative = np.where(manipulated_mask[0][x]==1, 0, 1)
            positive = np.where(manipulated_mask[0][x]==1, 255., 0)
            edit_im = img_orig.detach().cpu().numpy().copy()
            c = classes[x]
            edit_im[c] = edit_im[c]*negative + positive*(1-transp) + edit_im[c]*transp
            im = edit_im.squeeze().transpose(1,2,0)
            im = np.clip(im, 0, 1)

        return im  

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './hw3/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './hw3/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./hw3/data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./hw3/data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(11)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    # train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()


    # resnet50_fpn = Resnet50Backbone()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # solo_head = SOLOHead(num_classes=4,device=device) ## class number is 4, because consider the background as one category.
    # solo_head.to(device)
    # # loop the image
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for iter, data in enumerate(train_loader, 0):
    #     img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
    #     # fpn is a dict
    #     backout = resnet50_fpn(img)
    #     fpn_feat_list = list(backout.values())
    #     # make the target

    #     ## demo
    #     cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
    #     ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
    #                                                                      bbox_list,
    #                                                                      label_list,
    #                                                                      mask_list)
    #     mask_color_list = ["jet", "ocean", "Spectral"]
    #     solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
    #     break

    # model = SOLOTrainer()
    # model.backbone.load_state_dict(torch.load("./hw3/backbone.pth"))
    # model.solo_head.load_state_dict(torch.load("./hw3/solo.pth"))

    # model.eval()
    # count = 0
    # temp_flag = True
    # for i, batch_set in enumerate(test_loader):

    #     img_set  = batch_set[0]
    #     lab_set  = batch_set[1]
    #     mask_set = batch_set[2]
    #     bbox_set = batch_set[3]
    #     cat_pred, msk_pred = model.forward(img_set, eval=False)
    #     msk_tar, act_msk, cat_tar = model.solo_head.target(msk_pred,bbox_set, lab_set, mask_set)
    #     cat_pred, msk_pred = model.forward(img_set, eval=True)
        

    #     for i in range(batch_size):
    #         img_raw = img_set[i].squeeze(0)
    #         nms_ip = [model.solo_head.points_nms(cat_pred[j][i].unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1) for j in range(5)]
    #         cat_ip = [each_cat_level.squeeze(0) for each_cat_level in nms_ip]
    #         msk_ip = [msk_pred[j][i] for j in range(5)]
    #         fin_msk, fin_cls = model.solo_head.post_process(cat_ip, msk_ip)
    #         model.solo_head.visualize_nms_image(img_raw, fin_msk, fin_cls,thresh=0.2, transp=0.2)
    #         count += 1
    #         if count == 6:
    #             temp_flag = False
    #             break
            
    #     if temp_flag == False:
    #         break  



