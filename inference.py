
import torch
from trainer import SOLOTrainer
import matplotlib.pyplot as plt
from datasets.solo_dataset import BuildDataLoader, BuildDataset

def inference(device,test_loader, batch_size):
    model = SOLOTrainer()

    model.backbone.load_state_dict(torch.load('backbone.pth'))
    model.solo_head.load_state_dict(torch.load('solo.pth'))
    model.eval()

    model = model.cpu()
    model.backbone = model.backbone.cpu()
    model.solo_head = model.solo_head.cpu()

    count = 0
    temp_flag = True

    for i, batch_set in enumerate(test_loader):

        img_set  = batch_set[0]
        lab_set  = batch_set[1]
        mask_set = batch_set[2]
        bbox_set = batch_set[3]
        img_set = img_set.to(device)
        cat_pred, msk_pred = model.forward(img_set, eval=False)
        msk_tar, act_msk, cat_tar = model.solo_head.target(msk_pred,bbox_set, lab_set, mask_set)
        cat_pred, msk_pred = model.forward(img_set, eval=True)
        

        for i in range(batch_size):
            img_raw = img_set[i].squeeze(0)
            nms_ip = [model.solo_head.points_nms(cat_pred[j][i].unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1) for j in range(5)]
            cat_ip = [each_cat_level.squeeze(0) for each_cat_level in nms_ip]
            msk_ip = [msk_pred[j][i] for j in range(5)]
            fin_msk, fin_cls = model.solo_head.PostProcess(msk_ip, cat_ip)
            res_img = model.solo_head.visualize_nms_image(img_raw, fin_msk, fin_cls,thresh=0.5, transp=0.2)
            plt.figure(figsize=(20,20))
            plt.imshow(res_img)
            plt.show()
            count += 1
            if count == 6:
                temp_flag = False
                break

            del nms_ip, cat_ip, msk_ip
            del fin_msk, fin_cls


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset_solo = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset_solo)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(11)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_solo, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 1
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()
    inference(device,test_loader, batch_size)