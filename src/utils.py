from torchvision import transforms
import torch
from torch import nn

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
import pickle
from numpy import asarray
import os

from math import log10, sqrt
import cv2

def create_dataloaders(data_dir,batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    data_transforms = {
      'Training': train_transform,
      'Test_images': valid_transform
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['Training', 'Test_images']}
    # train_set, val_set = torch.utils.data.random_split(image_datasets['Training'], image_datasets['Val'])
    # dataset = {'train':train_set, 'val':val_set}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=1)
                for x in ['Training', 'Test_images']}

    return dataloaders

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True))

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def compress(image_file, quality, save_path):

    image = Image.open(image_file)

    image.save(save_path,
                 "JPEG",
                 optimize = True,
                 quality = quality)
    return

def convert_tensor_to_image(t, B=2, show_noise=True, epoch=None, model=None):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if show_noise:
        fig = plt.figure(figsize=(10, 7))
        # setting values to rows and column variables
        rows = 1
        columns = 3

    for i, img in enumerate(t):

        z = img * torch.tensor(std).view(3, 1, 1)
        z = z + torch.tensor(mean).view(3, 1, 1)
        img2 = transforms.ToPILImage(mode='RGB')(z)
        if epoch != None:
            model_path = f'dataset/images_during_training/{model}'
            epoch_path = f'dataset/images_during_training/{model}/epoch_{epoch}'

            if not os.path.exists(epoch_path):

                if not os.path.exists(model_path):
                    os.mkdir(model_path)

                os.mkdir(epoch_path)

            img2.save(epoch_path + f'/{i}.png')

        if show_noise:
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img2)
            plt.axis('off')
            plt.title(f"B={str(B)}")

def convert_one_tensor_to_one_image(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    z = t * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    img2 = transforms.ToPILImage(mode='RGB')(z)
    plt.imshow(img2)

    return img2

def convert_to_image(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    z = t * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    im = transforms.ToPILImage(mode='RGB')(z)

    return asarray(im), im

def compress_images_with_jpg():
    path = 'dataset/inference_images'
    s_p = 'dataset/inference_images/jpg_compress'

    main_dict = {}

    for f in os.listdir(path):
        f_path = os.path.join(path, f)

        if os.path.isfile(f_path) and ('.png' or '.jpg' in f):
            main_dict[str(f) + '_psnr'] = []
            main_dict[str(f) + '_bpp'] = []

            folder_name = f.split('.')[0]
            folder_path = os.path.join(s_p, folder_name)

            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            for quality in range(50, 101, 5):
                save_path = os.path.join(s_p, f.replace('.png', ''), f'quality_{quality}.jpeg')
                # print(save_path)
                compress(f_path, quality, save_path)

                original = cv2.imread(f_path)
                compressed = cv2.imread(save_path)
                psnr = PSNR(original, compressed)

                # Get sizes
                compressed_sz = os.path.getsize(save_path)
                # Get pixels product
                compressed_shape_multiple = compressed.shape[0] * compressed.shape[1]
                # Get bpp
                compressed_bpp = compressed_sz * 8 / compressed_shape_multiple

                main_dict[str(f) + '_psnr'].append(psnr)
                main_dict[str(f) + '_bpp'].append(compressed_bpp)

    return main_dict