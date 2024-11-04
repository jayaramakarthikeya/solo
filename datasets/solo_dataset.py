
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        self.images = h5py.File(path[0],'r')['data']
        self.masks = h5py.File(path[1],'r')['data']
        self.bboxes = np.load(path[3], allow_pickle=True)
        self.labels = np.load(path[2], allow_pickle=True)

        self.corresponding_masks = []
        # Aligning masks with labels
        count = 0
        for i, label in enumerate(self.labels):
            n = label.shape[0] 
            temp = []
            for j in range(n):
                temp.append(self.masks[count])
                count += 1
            self.corresponding_masks.append(temp)

        # Applying rescaling and mean, std, padding
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((800, 1066)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize = torchvision.transforms.Resize((800, 1066))
        self.pad = torch.nn.ZeroPad2d((11,11,0,0))

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__

        label = self.labels[index]
        transed_img = self.images[index]
        transed_bbox = self.bboxes[index]
        transed_mask = self.corresponding_masks[index]

        label = torch.tensor(label)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(transed_img, transed_mask, transed_bbox)

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    def __len__(self):
        return len(self.images)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, msks, box):
        # TODO: image preprocess

        img_normalized = img / 255.                     # Normalized between 0 & 1
        img_normalized = torch.tensor(img_normalized, dtype=torch.float)   # Converted to tensor
        img_scaled = self.transform(img_normalized)    # Rescaled to (800, 1066) and adjusted for given mean and std

        img_final = self.pad(img_scaled)              # Padded with zeros to get the shape as (800, 1088)

        msk_final = torch.zeros((len(msks), 800, 1088))           #Initializing mask tensor

        for i, msk in enumerate(msks):
            msk = msk/1.                                  # Converting it to uint8
            msk = torch.tensor(msk, dtype=torch.float).view(1,300,400)         # Converting it to tensor
            msk_scaled = self.pad(self.resize(msk).view(800,1066))            # Padding and resizing
            msk_scaled[msk_scaled < 0.5] = 0
            msk_scaled[msk_scaled > 0.5] = 1
            msk_final[i] = msk_scaled

        box = torch.tensor(box, dtype=torch.float)
        box_final = torch.zeros_like(box)
        box_final[:,0] = box[:,0] * 800/300                  # Scaling x
        box_final[:,2] = box[:,2] * 800/300                  # Scaling x
        box_final[:, 1] = box[:,1] * (1066/400) + 11                # Scaling y
        box_final[:, 3] = box[:,3] * (1066/400) + 11                # Scaling y

        # check flag
        assert img_final.shape == (3, 800, 1088)

        return img_final, msk_final, box_final

    def visualize_raw_processor(self,img, label, mask, bbox, alpha=0.5):
        processed_mask = mask.clone().detach().squeeze().bool()
        img = img.clone().detach()[:, :, 11:-11]

        inv_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[ 0., 0., 0. ], std=[1/0.229, 1/0.224, 1/0.255]), torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
        pad = torch.nn.ZeroPad2d((11,11,0,0))
        processed_img = pad(inv_transform(img))

        processed_img = processed_img.numpy()

        processed_img = np.clip(processed_img, 0, 1)
        processed_img = torch.from_numpy((processed_img * 255.).astype(np.uint8))

        img_to_draw = processed_img.detach().clone()

        if processed_mask.ndim == 2:
            processed_mask = processed_mask[None, :, :]
        for mask, box, lab in zip(processed_mask, bbox, label):
            if lab.item() == 1 :            # vehicle
                colored = 'blue'
                box = box.numpy().astype(np.uint8)
                color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
                img_to_draw[:, mask] = color[:, None]
            if lab.item() == 2 :            # person
                colored = 'green'
                box = box.numpy().astype(np.uint8)
                color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
                img_to_draw[:, mask] = color[:, None]
            if lab.item() == 3 :            # animal
                colored = 'red'
                box = box.numpy().astype(np.uint8)
                color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
                img_to_draw[:, mask] = color[:, None]
    
        out = (processed_img * (1 - alpha) + img_to_draw * alpha).to(torch.uint8)
        out = draw_bounding_boxes(out, bbox, colors='red', width=2)
        final_img = out.numpy().transpose(1,2,0)
        return final_img


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn

        images = []
        labels = []
        masks = []
        bboxes = []

        for i in batch:
            images.append(i[0])
            labels.append(i[1])
            masks.append(i[2])
            bboxes.append(i[3])

        return torch.stack(images, dim=0), labels, masks, bboxes

    def loader(self):
        # TODO: return a dataloader
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                          num_workers=self.num_workers, collate_fn=self.collect_fn)



## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './hw3/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './hw3/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './hw3/data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './hw3/data/hw3_mycocodata_bboxes_comp_zlib.npy'
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
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()



    # for iter, data in enumerate(train_loader, 0):
    #     img, label, mask, bbox = [data[i] for i in range(len(data))]
    #     # check flag
    #     plt.figure()
    #     plt.imshow(img[0].permute(1, 2, 0))
    #     plt.show()
    #     assert img.shape == (batch_size, 3, 800, 1088)
    #     assert len(mask) == batch_size
    #     break

    # mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for iter, data in enumerate(train_loader, 0):

    #     img, label, mask, bbox = [data[i] for i in range(len(data))]
    #     # check flag
    #     assert img.shape == (batch_size, 3, 800, 1088)
    #     assert len(mask) == batch_size

    #     label = [label_img.to(device) for label_img in label]
    #     mask = [mask_img.to(device) for mask_img in mask]
    #     bbox = [bbox_img.to(device) for bbox_img in bbox]


    #     # plot the origin img
    #     for i in range(batch_size):
    #         ## TODO: plot images with annotations
    #         plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
    #         plt.show()

    #     if iter == 10:
    #         break
    plt.figure(figsize=(20, 20))
    #fig, ax = plt.subplots(5, figsize=(10, 10))
    count = 0
    temp_flag = True

    for i, batch_set in enumerate(train_loader):
        img_set = batch_set[0]
        lab_set = batch_set[1]
        mask_set = batch_set[2]
        bbox_set = batch_set[3]
        for single_img, single_lab, single_mask, single_bbox in zip(img_set, lab_set, mask_set, bbox_set):
            final_img = dataset.visualize_raw_processor(single_img, single_lab, single_mask, single_bbox)
            plt.subplot(1, 5, count + 1)
            plt.imshow(final_img)
            count += 1
            if count == 5:
                temp_flag = False
                break
        if temp_flag == False:
            break

    plt.show()