
import torch
import matplotlib.pyplot as plt
from datasets.solo_dataset import BuildDataset
from models.backbone import Resnet50Backbone
from models.solo_head import SOLOHead

def plot_dataset(train_loader,dataset:BuildDataset):
    plt.figure(figsize=(20, 20))
    #fig, ax = plt.subplots(5, figsize=(10, 10))
    count = 0
    temp_flag = True
    torch.manual_seed(20)
    for i, batch_set in enumerate(train_loader):
        img_set = batch_set[0]
        lab_set = batch_set[1]
        mask_set = batch_set[2]
        bbox_set = batch_set[3]
        for single_img, single_lab, single_mask, single_bbox in zip(img_set, lab_set, mask_set, bbox_set):
            final_img = dataset.visualize_raw_processor(single_img, single_lab, single_mask, single_bbox)
            plt.subplot(1, 5, count + 1)
            plt.imshow(final_img)
            plt.axis('off')
            count += 1
            if count == 5:
                temp_flag = False
                break
        if temp_flag == False:
            break

    plt.show()

def plot_fpn(train_loader):
    resnet50_fpn = Resnet50Backbone()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solo_head = SOLOHead(num_classes=4,device=device) ## class number is 4, because consider the background as one category.
    solo_head.to(device)
    # loop the image
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                            bbox_list,
                                                                            label_list,
                                                                            mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
        if iter == 2:
            break

        

