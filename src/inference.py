import torch
from torchvision import transforms
import os
import re
import sys
import pickle

import argparse
from PIL import Image

from utils import convert_one_tensor_to_one_image, PSNR, compress_images_with_jpg, convert_to_image
import matplotlib.pyplot as plt

from train import initialize_models
from model import encoder, decoder, MobileNetV3, h_swish, h_sigmoid, InvertedResidual, SELayer, decoder_Mobilenet, feature_extractor_1x1

valid_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

parser = argparse.ArgumentParser(description="Predictor")
#parser.add_argument("-m", "--model", type=str, help="Choose model from Mobilenet, Uresnet, VGG", required=True, default='VGG')
parser.add_argument("-mp", "--model_path", type=str, help="You should specify path to your model weights",
                    required=False, default='Weights/pretrained_for_inference_weights/Mobilenet')
parser.add_argument("-b", "--B", type=int, help="Extent of compression",
                    required=False, default=4)
parser.add_argument("-ip", "--images_path", type=str, help="Write images folder",
                    required=False, default='dataset/inference_images')
parser.add_argument("-ibg", "--is_build_graph", type=str, help="Specify the main weights path if you want to build a graph",
                    required=False, default=None)
parser.add_argument("-cwp", "--choose_what_plot", type=str, help="You could specify name of image to compare PSNR/bpp between model and jpg compression",
                    required=False, default='miles_morales.png')
parser.add_argument("-poi", "--path_one_image", type=str, help="Compress-decompress one image and see result with chosen model path",
                    required=False, default='dataset/inference_images/miles_morales.png')
parser.add_argument("-oe", "--only_encode", type=str, help="Only encode means save compressed variant",
                    required=False, default='True')

if __name__ == "__main__":

    args = parser.parse_args()
    images_path = args.images_path
    is_build_graph = args.is_build_graph

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_build_graph!=None:
        assert len(os.listdir(f'{is_build_graph}/Encoder')) >= 3 and len(os.listdir(f'{is_build_graph}/Decoder')), 'you must have 3 or more different weights in order to build a graph'

        Compression, train = True, False
        main_dict = {}
        model = args.is_build_graph.split('/')[-1]
        for f in os.listdir(images_path):
            f_path = os.path.join(images_path, f)
            if os.path.isfile(f_path) and ('.png' or '.jpg' in f_path):

                main_dict['autoencoder_' + str(f) + '_psnr'] = []
                main_dict['autoencoder_' + str(f) + '_bpp'] = []
                main_dict['autoencoder_' + str(f) + '_compressed_images'] = []

                f_s = f'dataset/inference_images/autoencoder_compressed/{model}'
                if not os.path.exists('dataset/inference_images/autoencoder_compressed'):
                    os.mkdir('dataset/inference_images/autoencoder_compressed')
                if not os.path.exists(f_s):
                    os.mkdir(f_s)

                name_models = os.listdir(f'{is_build_graph}/Encoder')
                for epoch_B in name_models:

                    pattern = r'B_\d+'
                    sp = re.findall(pattern, epoch_B)
                    element = sp[0]
                    B = int(element[2:])
                    folder_name = f.split('.')[0]
                    folder_path = os.path.join(f_s, folder_name)

                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    save_path = f'{folder_path}/{B}.bin'

                    enc, dec, feat_extract = initialize_models(model=model, inference_path=is_build_graph,epoch_B=epoch_B)
                    # print(enc_path)

                    input_image = Image.open(f_path)
                    input_tensor = valid_transform(input_image)
                    input_im = input_tensor.unsqueeze(0).to(device)

                    outputs = enc(input_im, B, Compression=True, train=False, save_path=save_path, feat_extract=feat_extract)
                    outs = dec.decompressed(save_path=save_path)
                    x = dec(outs)

                    compressed_im, c_IM = convert_to_image(x[0])
                    original_im, o_IM = convert_to_image(input_im[0])

                    psnr = PSNR(original_im, compressed_im)

                    # Get sizes
                    compressed_sz = os.path.getsize(save_path) * 8
                    print('Filename - ', f, ', Size - ', compressed_sz, 'бит, параметр B - ', B)
                    # Get pixels product
                    compressed_shape_multiple = compressed_im.shape[0] * compressed_im.shape[1]
                    # Get bpp
                    compressed_bpp = compressed_sz / compressed_shape_multiple

                    main_dict['autoencoder_' + str(f) + '_psnr'].append(psnr)
                    main_dict['autoencoder_' + str(f) + '_bpp'].append(compressed_bpp)
                    main_dict['autoencoder_' + str(f) + '_compressed_images'].append(c_IM)

        jpg_main_dict = compress_images_with_jpg()
        # plotting the points
        choose_what_plot = args.choose_what_plot
        f = choose_what_plot
        plt.plot(main_dict['autoencoder_' + str(f) + '_bpp'], main_dict['autoencoder_' + str(f) + '_psnr'], color='r', label=f'{f} by autoencoder')
        plt.plot(jpg_main_dict[str(f) + '_bpp'], jpg_main_dict[str(f) + '_psnr'], color='g', label=f'{f} by python jpg')

        # naming the x axis
        plt.xlabel('bpp')
        # naming the y axis
        plt.ylabel('psnr')

        # giving a title to my graph
        plt.title('BPP/PSNR result')
        leg = plt.legend(loc='lower right')
        if model == 'Uresnet':
            plt.xlim(2790, 2815)
            plt.ylim(0, 45)
        elif model=='Mobilenet':
            plt.xlim(0, 40)
            plt.ylim(27, 29)
        else:
            plt.xlim(0, 40)
            plt.ylim(0, 45)

        # function to show the plot
        plt.show()

    else:
        model_path = args.model_path
        B = args.B
        path_image = args.path_one_image
        only_encode = args.only_encode

        model = model_path.split('/')[-1]
        for pt in os.listdir(f'{model_path}/Encoder'):
            if str(B) in pt:
                epoch_B = pt
        enc, dec, feat_extract = initialize_models(model=model, inference_path=model_path, epoch_B=epoch_B)

        input_image = Image.open(path_image)
        input_tensor = valid_transform(input_image)
        input_im = input_tensor.unsqueeze(0).to(device)

        path_folder = f'dataset/inference_images/save_compressed_images/{model}'

        if not os.path.exists('dataset/inference_images/save_compressed_images'):
            os.mkdir('dataset/inference_images/save_compressed_images')
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        save_path = f'{path_folder}/encoded_my_image_B_{B}.bin'

        if only_encode=='True':

            outputs = enc(input_im, B=B, Compression=True, train=False, save_path=save_path, feat_extract=feat_extract)
            sys.exit(f'Image is encoded at path {save_path}')

        else:

            outs = dec.decompressed(save_path=save_path)
            x = dec(outs)

            save_decode = f'{path_folder}/decoded_my_image_B_{B}.jpeg'
            print(x.shape)
            img = convert_one_tensor_to_one_image(x[0].to('cpu'))
            img.save(save_decode)
            print(f'Image is decoded at path {save_decode}')







