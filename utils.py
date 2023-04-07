import glob
from easydict import EasyDict as edict

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vision_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def VQVAE_cfg():
    __C                                              = edict()
    cfg                                              = __C


    __C.DATASET                                      = edict()
    __C.DATASET.DATA_DIR                             = '/home/ml/rajan/gans/5.VQVAE/celeba-dataset'
    __C.DATASET.IMAGE_CHANNELS                       = 3
    __C.DATASET.NUM_WORKERS                          = 8
    __C.DATASET.BATCH_SIZE                           = 128
    __C.DATASET.IMAGE_SIZE                           = 128
    __C.DATASET.NUM_OF_IMAGES                        = -1
    
    __C.MODEL                                        = edict()
    __C.MODEL.NUM_DOWNSAMPLINGS                      = 5
    __C.MODEL.ENCODER_CHANNELS                       = 64
    __C.MODEL.LATENT_CHANNELS                        = 64
    __C.MODEL.NUM_EMBEDDINGS                         = 256


    __C.TRAIN                                        = edict()
    __C.TRAIN.EPOCHS                                 = 20
    __C.TRAIN.LEARNING_RATE                          = 1e-2
    __C.TRAIN.WEIGHT_DECAY                           = 1e-2
    __C.TRAIN.DEVICE                                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    return cfg

cfg = VQVAE_cfg()


class CelebDataset(Dataset):
    def __init__(self, paths, split) -> None:
        super().__init__()
        self.items = paths
        self.split = split
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(cfg.DATASET.IMAGE_SIZE),
                transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(cfg.DATASET.IMAGE_SIZE),
                transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img = Image.open(self.items[index % len(self.items)]).convert("RGB")
        if self.split == "train":
            return self.train_transform(img)
        else:
            return self.val_transform(img)



def get_data_loaders():
    images_dir = glob.glob(
    "celeba-dataset/img_align_celeba/img_align_celeba/*.jpg"
    )

    images_dir = images_dir[0:cfg.DATASET.NUM_OF_IMAGES]
    train_paths, test_paths = train_test_split(
        images_dir,
        test_size=0.02,
        random_state=42,
    )

    train_loader = DataLoader(
        CelebDataset(train_paths, "train"),
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )
    val_loader = DataLoader(
        CelebDataset(test_paths, "test"),
        batch_size=int(cfg.DATASET.BATCH_SIZE * 0.75),
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )
    
    return train_loader, val_loader

def plot_batch(ax, batch, title=None, **kwargs):
    imgs = vision_utils.make_grid(batch, padding=2, normalize=True)
    imgs = np.moveaxis(imgs.numpy(), 0, -1)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax.imshow(imgs, **kwargs)


def show_images(batch, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_batch(ax, batch, title)
    # plt.savefig("foo.png")
    plt.show()


def reconstruct(model, batch, device):
    batch = batch.to(device)
    with torch.no_grad():
        reconstructed_batch = model(batch)[0]
    reconstructed_batch = reconstructed_batch.cpu()
    return reconstructed_batch


def show_2_batches(batch1, batch2, title1, title2,epoch):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    plot_batch(ax, batch1, title1)

    ax = fig.add_subplot(122)
    plot_batch(ax, batch2, title2)
    # plt.show()
    plt.savefig(f'result/plot_{str(epoch)}.png')
    


def plot_history_train_val(history, key):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.arange(1, len(history['train ' + key]) + 1)
    ax.plot(xs, history['train ' + key], '.-', label='train')
    ax.plot(xs, history['val ' + key], '.-', label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel(key)
    ax.grid()
    ax.legend()
    # plt.show()
    plt.savefig(f"result/{key}.png")