import torch
from torch import nn
import random
import os

from model import Uresnet18, encoder, decoder, VGGencoder, VGGdecoder, VGGauthoencoder, mobilenetv3_small, decoder_Mobilenet, Mobilenet_authoencoder
from utils import create_dataloaders, convert_tensor_to_image

import argparse
import lpips

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description="Predictor")
parser.add_argument("-b", "--B", type=int, help="Specify B number ", required=True, default=2)
parser.add_argument("-m", "--model", type=str, help="Choose model from Mobilenet, Uresnet, VGG", required=True, default='VGG')
parser.add_argument("-wp", "--weights_path", type=str, help="Specify weights path", required=True, default=None)
parser.add_argument("-r", "--is_random", type=bool, help="Specify if B is constant or random",
                    required=False, default=True)
parser.add_argument("-dp", "--dataset_path", type=str, help="Specify dataset directory",
                    required=False, default='dataset')
parser.add_argument("-e", "--epochs", type=int, help="Choose epochs numbers", required=False, default=3)
parser.add_argument("-al", "--alpha", type=float, help="Choose alpha for customloss", required=False, default=0.9)
parser.add_argument("-bl", "--beta", type=float, help="Choose beta for customloss", required=False, default=0.1)
parser.add_argument("-lr", "--learning_rate", type=float, help="Choose learning_rate", required=False, default=0.005)
parser.add_argument("-mm", "--momentum", type=float, help="Choose momentum", required=False, default=0.9)
parser.add_argument("-bs", "--batch_size", type=int, help="Choose batch_size", required=False, default=1)
parser.add_argument("-pf", "--print_frequency", type=int,
                    help="Choose how frequent to print test logs in train mode(i% --print_frequency)",
                    required=False, default=1)
parser.add_argument("-see", "--save_every_epoch", type=int,
                    help="Choose how frequent to save weights, after which epoch (epoch% --save_every_epoch)",
                    required=False, default=5)
parser.add_argument("-pe", "--plot_every", type=int, help="How often to save test result during train", required=False, default=1)



class CustomLoss(nn.Module):

    def __init__(self, alpha, beta):
        super(CustomLoss, self).__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        d = self.loss_fn_vgg(output[0], target[0])

        criterion = nn.MSELoss()
        loss = criterion(output, target)
        for i in range(1, output.shape[0]):
            d += self.loss_fn_vgg(output[i], target[i])

        loss= loss * self.alpha
        d = (d.item() * self.beta) / target.shape[0]
        print('Perceptual loss - ', d, 'MSE loss - ', loss)
        return loss + d


def train_EPOCHS(EPOCHS, What_B, dataloaders, save_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/compressing_trainer_{}'.format(timestamp))
    for epoch in range(EPOCHS):
        if epoch % 1 == 0:
            print('-' * 50, 'EPOCH {}:'.format(epoch + 1), '-' * 50)

        # Make sure gradient tracking is on, and do a pass over the data
        autoencoder.train(True)
        avg_loss = train_one_epoch(epoch, writer, What_B, dataloaders)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        autoencoder.eval()
        print('-' * 20, 'Val losses per validation batch', '-' * 20)
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            # for i, vdata in enumerate(validation_loader):
            for i, data in enumerate(dataloaders['Test_images']):

                flag, B = What_B
                if flag:
                    B = B
                else:
                    B = random.randint(2, 9)

                # Every data instance is an input + label pair
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                voutputs = autoencoder(inputs, B)
                if epoch % args.plot_every == 0:
                    convert_tensor_to_image(voutputs.to('cpu'), B=B, epoch=epoch, model=args.model)
                vloss = loss_fn(voutputs, inputs)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training', {'Training': avg_loss}, epoch)
        writer.flush()
        if epoch % args.save_every_epoch == 0:
            for f in ['Encoder', 'Decoder', 'Autoencoder']:
                folder_path = os.path.join(save_path, f)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

            if What_B[0]:
                B = What_B[1]
                enc_save_path = f'{save_path}/Encoder/B_{B}_epochs_{epoch + 1}.pt'
                dec_save_path = f'{save_path}/Decoder/B_{B}_epochs_{epoch + 1}.pt'
                aut_save_path = f'{save_path}/Autoencoder/B_{B}_epochs_{epoch + 1}.pt'

            else:
                enc_save_path = f'{save_path}/Encoder/B_random_epochs_{epoch + 1}.pt'
                dec_save_path = f'{save_path}/Decoder/B_random_epochs_{epoch + 1}.pt'
                aut_save_path = f'{save_path}/Autoencoder/B_random_epochs_{epoch + 1}.pt'

            torch.save(enc, enc_save_path)
            torch.save(dec, dec_save_path)
            torch.save(autoencoder, aut_save_path)

    return writer

def train_one_epoch(epoch_index, tb_writer, What_B, dataloaders):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloaders['Training']):

        flag, B = What_B
        if flag:
            B = B
        else:
            B = random.randint(2, 9)

        # Every data instance is an input + label pair
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = autoencoder(inputs, B)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, inputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % args.print_frequency == 0:
            last_loss = running_loss/args.print_frequency
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloaders['Training']) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def initialize_models(model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), inference_path=None, epoch_B=None):
    if model == 'Uresnet':
        enc = encoder().to(device)
        dec = decoder().to(device)
        autoencoder = Uresnet18(enc, dec).to(device)

    elif model == 'VGG':
        enc = VGGencoder().to(device)
        dec = VGGdecoder(enc.encoder).to(device)
        autoencoder = VGGauthoencoder(enc, dec).to(device)

    elif model == 'Mobilenet':
        enc = mobilenetv3_small()
        enc.load_state_dict(torch.load('Weights/mobilenet_v3_small-047dcff4.pth'))
        enc = enc.to(device)
        dec = decoder_Mobilenet().to(device)
        autoencoder = Mobilenet_authoencoder(enc, dec).to(device)

    if inference_path != None:

        enc_path = os.path.join(inference_path, 'Encoder', epoch_B)
        dec_path = os.path.join(inference_path, 'Decoder', epoch_B)

        enc = torch.load(enc_path, map_location=device)
        dec = torch.load(dec_path, map_location=device)

        return enc, dec

    return enc, dec, autoencoder

if __name__ == "__main__":

    args = parser.parse_args()
    What_B = [args.is_random, args.B]
    weights_path = args.weights_path
    model = args.model
    epochs = args.epochs

    if not os.path.exists(weights_path):
        os.mkdir((weights_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = create_dataloaders(args.dataset_path, batch_size=args.batch_size)
    enc, dec, autoencoder = initialize_models(model=model)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=args.learning_rate, momentum=args.momentum)
    loss_fn = CustomLoss(alpha=args.alpha, beta=args.beta)

    train_EPOCHS(EPOCHS=epochs, What_B=What_B, dataloaders=dataloaders, save_path=weights_path)