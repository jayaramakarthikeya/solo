
import pytorch_lightning as pl
import torchvision
from models.solo_head import SOLOHead
from pytorch_lightning import loggers as pl_loggers
from torch.optim.lr_scheduler import MultiStepLR
import torch
import datasets.solo_dataset as dataset
import matplotlib.pyplot as plt
from models.backbone import Resnet50Backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SOLOTrainer(pl.LightningModule):
    _default_cfg = {
        'num_classes': 4,
        'in_channels': 256,
        'seg_feat_channels': 256,
        'stacked_convs': 7,
        'strides': [8, 8, 16, 32, 32],
        'scale_ranges': [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        'epsilon': 0.2,
        'num_grids': [40, 36, 24, 16, 12],
        'mask_loss_cfg': dict(weight=3),
        'cate_loss_cfg': dict(gamma=2, alpha=0.25, weight=1),
        'postprocess_cfg': dict(cate_thresh=0.2, mask_thresh=0.5, pre_NMS_num=50, keep_instance=5, IoU_thresh=0.5)
    }

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)

        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = pretrained_model.backbone
        self.solo_head = SOLOHead(self.num_classes, device)
        
        self.train_outputs = []
        self.train_loss_epoch = []
        self.train_cat_loss_epoch = []
        self.train_msk_loss_epoch = []
        
        self.validation_outputs = []
        self.val_loss_epoch = []
        self.val_cat_loss_epoch = []
        self.val_msk_loss_epoch = []


    # Forward function should calculate across each level of the feature pyramid network.
    # Input:
    #     images: batch_size number of images
    # Output:
    #     if eval = False
    #         category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
    #     if eval==True
    #         category_predictions: list, len(fpn_levels), each (batch_size, S, S, C-1)
    #         / after point_NMS
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, image_h/4, image_w/4)
    #         / after upsampling
    def forward(self, images, eval=True):
        # you can modify this if you want to train the backbone
        feature_pyramid = [v.detach() for v in self.backbone(images).values()] # this has strides [4,8,16,32,64]
        
        cate_pred_list, ins_pred_list = self.solo_head(feature_pyramid, eval=eval)

        return cate_pred_list, ins_pred_list


    def training_step(self, batch, batch_idx):
      images, labels, masks, bounding_boxes = batch
      images = images.to(device)
      cat_pred, msk_pred = self.forward(images, eval=False)
      mask_targets, active_masks, category_targets = self.solo_head.target(msk_pred,bounding_boxes, labels, masks)
      cat_loss, msk_loss, train_loss = self.solo_head.loss(cat_pred, msk_pred, mask_targets, active_masks, category_targets)

      # total_memory = torch.cuda.get_device_properties(0).total_memory
      # reserved_memory = torch.cuda.memory_reserved(0)
      # allocated_memory = torch.cuda.memory_allocated(0)
      # free_memory = reserved_memory - allocated_memory
      
      # Print memory details
      # print(f"Device: {device}")
      # print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
      # print(f"Available Memory: {free_memory / (1024 ** 3):.2f} GB")
  
      # del images, labels, masks, bounding_boxes
      # del mask_targets, category_targets, active_masks
      torch.cuda.empty_cache()

      # total_memory = torch.cuda.get_device_properties(0).total_memory
      # reserved_memory = torch.cuda.memory_reserved(0)
      # allocated_memory = torch.cuda.memory_allocated(0)
      # free_memory = reserved_memory - allocated_memory
      
      # # Print memory details
      # print(f"Device: {device}")
      # print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
      # print(f"Available Memory: {free_memory / (1024 ** 3):.2f} GB")

      self.log("train_loss", train_loss.detach().cpu().item(), prog_bar=True, on_epoch=True)
      self.log("mask_loss", msk_loss.detach().cpu().item(), prog_bar=True)
      self.log("category loss", cat_loss.detach().cpu().item(), prog_bar=True)

      self.train_outputs.append({"loss":train_loss.detach().cpu().item(), "msk_loss":msk_loss.detach().cpu().item(), "cat_loss":cat_loss.cpu().item()})
      return train_loss

    def on_train_epoch_end(self):
      avg_train_loss = 0
      avg_cat_loss = 0
      avg_mask_loss = 0
      for i in range(len(self.train_outputs)):
        avg_train_loss += self.train_outputs[i]["loss"]
        avg_cat_loss += self.train_outputs[i]["cat_loss"]
        avg_mask_loss += self.train_outputs[i]["msk_loss"]
      
      self.train_loss_epoch.append(avg_train_loss)
      self.train_cat_loss_epoch.append(avg_cat_loss)
      self.train_msk_loss_epoch.append(avg_mask_loss)

      self.train_outputs = []


    def validation_step(self, batch, batch_idx):
      images, labels, masks, bounding_boxes = batch

      images = images.to(device)
      cat_pred, msk_pred = self.forward(images, eval=False)
      mask_targets, active_masks, category_targets = self.solo_head.target(msk_pred,bounding_boxes, labels, masks)
      cat_loss, msk_loss, val_loss = self.solo_head.loss(cat_pred, msk_pred, mask_targets, active_masks, category_targets)

      #del images, labels, masks, bounding_boxes
      #del mask_targets, category_targets, active_masks
      torch.cuda.empty_cache()

      self.log("val_loss", val_loss.detach().cpu().item(),prog_bar=True,on_epoch=True)
      self.validation_outputs.append({"loss":val_loss.detach().cpu().item(), "msk_loss":msk_loss.detach().cpu().item(), "cat_loss":cat_loss.detach().cpu().item()})

    def on_validation_epoch_end(self):
      avg_train_loss = 0
      avg_cat_loss = 0
      avg_mask_loss = 0
      for i in range(len(self.validation_outputs)):
        avg_train_loss += self.validation_outputs[i]["loss"]
        avg_cat_loss += self.validation_outputs[i]["cat_loss"]
        avg_mask_loss += self.validation_outputs[i]["msk_loss"]
      
      self.val_loss_epoch.append(avg_train_loss)
      self.val_cat_loss_epoch.append(avg_cat_loss)
      self.val_msk_loss_epoch.append(avg_mask_loss)

      self.validation_outputs = []


    def configure_optimizers(self):
      optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
      scheduler = MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)

      return optimizer

if __name__ == "__main__":
    imgs_path = './hw3/data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './hw3/data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './hw3/data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './hw3/data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset_solo = dataset.BuildDataset(paths)

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
    train_build_loader = dataset.BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    train_loader = train_build_loader.loader()
    test_build_loader = dataset.BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = test_build_loader.loader()

    model = SOLOTrainer()
    #model.backbone.load_state_dict(torch.load("./hw3/backbone.pth"))
    #model.solo_head.load_state_dict(torch.load("./hw3/solo.pth"))
    epochs = 5

    logger = pl_loggers.TensorBoardLogger("tb_logs", name="SOLO")
    # trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=epochs, logger=logger)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, logger=logger)
    trainer.fit(model, train_loader, test_loader)

    torch.save(model.solo_head.state_dict(), "./solo_head.pth")
    torch.save(model.backbone.state_dict(), "./backbone.pth")

    tot_cat_train_losses = model.train_cat_loss_epoch
    plt.plot(tot_cat_train_losses)
    plt.title("Training Category loss (Focal loss) per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    tot_msk_train_losses = model.train_msk_loss_epoch
    plt.plot(tot_msk_train_losses)
    plt.title("Training Mask loss (Dice loss) per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    tot_train_losses = model.train_loss_epoch
    plt.plot(tot_train_losses)
    plt.title("Training loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    tot_cat_val_losses = model.val_cat_loss_epoch
    plt.plot(tot_cat_val_losses)
    plt.title("Validation Category loss (Focal loss) per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    tot_msk_val_losses = model.val_msk_loss_epoch
    plt.plot(tot_msk_val_losses)
    plt.title("Validation Mask loss (Dice loss) per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    tot_val_losses = model.val_loss_epoch
    plt.plot(tot_val_losses)
    plt.title("Validation loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()