import torch
import torchvision
from torch import nn
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel
import random

import pickle

import collections

from utils import convrelu
import math


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        # self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, B, Compression=False, train=True,
                save_path=None):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.sigmoid(layer4)
        if train:
            layer4 = layer4 + 1 / (2 ** B) * random.uniform(-0.5, 0.5)
        if not train:
            layer4 = torch.round(layer4 * (2 ** B))
            layer4 = layer4.type(torch.int8)

        if Compression:
            for_compress = torch.flatten(layer4)
            length = for_compress.shape[0]
            for_compress = for_compress.tolist()
            frequency = collections.Counter(for_compress)
            for k in frequency.keys():
                frequency[k] = frequency[k] / len(for_compress)
            model = SimpleAdaptiveModel(frequency)
            coder = AECompressor(model)
            compressed = coder.compress(for_compress)

            if save_path != None:
                torch_compressed = torch.tensor(compressed, dtype=torch.int8)

                outs = [layer4, layer3, layer2, layer1, layer0, x_original]
                list_to_save = [torch_compressed, outs, coder, length, B]

                with open(save_path, "wb") as fp:  # Pickling
                    pickle.dump(list_to_save, fp)

        outs = [layer4, layer3, layer2, layer1, layer0, x_original]

        return outs


class decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 3, 1)

    def decompressed(self, save_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if save_path != None:
            with open(save_path, 'rb') as pf:
                outputs = pickle.load(pf)
            pf.close()

        x, outs, coder, length, B = outputs
        x_list = x.tolist()

        decompressed_list = coder.decompress(x_list, length_encoded=length)
        decompressed_tensor = torch.FloatTensor(decompressed_list)
        x = torch.reshape(decompressed_tensor, [-1,512,7,7]).to(device)
        x = x / (2 ** B)
        outs.pop(0)
        outs.insert(0, x)
        return outs

    def forward(self, outs):
        layer4, layer3, layer2, layer1, layer0, x_original = outs

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class Uresnet18(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, B, Compression=False, train=True, save_path=None):
        outputs = self.encoder(x, B, Compression=Compression,
                               train=train, save_path=save_path)

        if Compression:
            outputs = self.decoder.decompressed(save_path=save_path)

        x = self.decoder(outputs)

        return x


class VGGencoder(nn.Module):
    '''Encoder of image based on the architecture of VGG-16 with batch normalization.

    Args:
        pretrained_params (bool, optional): If the network should be populated with pre-trained VGG parameters.
            Defaults to True.

    '''
    channels_in = 3
    channels_code = 512

    def __init__(self, pretrained_params=True):
        super(VGGencoder, self).__init__()

        vgg = torchvision.models.vgg11_bn(pretrained=pretrained_params)

        del vgg.avgpool
        del vgg.classifier

        self.encoder = self._encodify_(vgg)

    def forward(self, x, B, Compression=False, train=True, save_path=None):
        '''Execute the encoder on the image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_code (Tensor): code tensor
            pool_indices (list): Pool indices tensors in order of the pooling modules

        '''
        pool_indices = []
        x_current = x
        for module_encode in self.encoder:

            if isinstance(module_encode, nn.Linear) and module_encode.in_features == 25088:
              x_current = torch.reshape(x_current, (1, 25088))

            output = module_encode(x_current)

            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output
        # Noize
        if train:
          x_current = x_current + 1/2**B * random.uniform(-0.5, 0.5)

        if Compression:
          batch_size = x_current.shape[0]
          x_current = torch.round(x_current * 2**B)
          for_compress = torch.flatten(x_current)
          for_compress = for_compress.tolist()
          frequency = collections.Counter(for_compress)

          for k in frequency.keys():
            frequency[k] = frequency[k]/len(for_compress)

          model = SimpleAdaptiveModel(frequency)
          coder = AECompressor(model)
          compressed = coder.compress(for_compress)

          if save_path != None:
            length = len(for_compress)
            torch_compressed = torch.tensor(compressed, dtype=torch.int8)
            list_to_save = [torch_compressed, coder, batch_size, length, B]

            with open(save_path, "wb") as fp:   #Pickling
              pickle.dump(list_to_save, fp)

        return x_current

    def _encodify_(self, encoder):
        '''Create list of modules for encoder based on the architecture in VGG template model.

        In the encoder-decoder architecture, the unpooling operations in the decoder require pooling
        indices from the corresponding pooling operation in the encoder. In VGG template, these indices
        are not returned. Hence the need for this method to extent the pooling operations.

        Args:
            encoder : the template VGG model

        Returns:
            modules : the list of modules that define the encoder corresponding to the VGG model

        '''
        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)
            else:
                modules.append(module)

            if isinstance(module, nn.AdaptiveAvgPool2d):
                module_add = nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          return_indices=True)

        try:
          modules.append(encoder.avgpool)
          for m in encoder.classifier:
            modules.append(m)
        except AttributeError:
          pass

        return modules


class VGGdecoder(torch.nn.Module):
    '''Decoder of code based on the architecture of VGG with batch normalization.

    The decoder is created from a pseudo-inversion of the encoder based on VGG with batch normalization. The
    pesudo-inversion is obtained by (1) replacing max pooling layers in the encoder with max un-pooling layers with
    pooling indices from the mirror image max pooling layer, and by (2) replacing 2D convolutions with transposed
    2D convolutions. The ReLU and batch normalization layers are the same as in the encoder, that is subsequent to
    the convolution layer.

    Args:
        encoder: The encoder instance of `EncoderVGG` that is to be inverted into a decoder

    '''
    channels_in = 3
    channels_out = 3

    def __init__(self, encoder):
        super(VGGdecoder, self).__init__()

        self.decoder = self._invert_(encoder)

    def decompressed(self, save_path=None):

        if save_path != None:
            with open(save_path, 'rb') as pf:
                list_to_save = pickle.load(pf)
            pf.close()

        torch_compressed, coder, batch_size, length, B = list_to_save
        torch_compressed_list = torch_compressed.tolist()
        decompressed_list = coder.decompress(torch_compressed_list, length_encoded=length)
        decompressed_tensor = torch.FloatTensor(decompressed_list).to(device)
        decompressed_tensor = decompressed_tensor / 2 ** B

        x = decompressed_tensor.view(batch_size, 512, 7, 7)

        return x

    #def forward(self, x, pool_indices):
    def forward(self, x):
        '''Execute the decoder on the code tensor input

        Args:
            x (Tensor): code tensor obtained from encoder
            pool_indices (list): Pool indices Pytorch tensors in order the pooling modules in the encoder

        Returns:
            x (Tensor): decoded image tensor

        '''
        x_current = x

        for module_decode in self.decoder:
            x_current = module_decode(x_current)

            if isinstance(module_decode, nn.Linear) and module_decode.out_features == 25088:
                x_current = torch.reshape(x_current, (x.shape[0], 512, 7, 7))

        return x_current

    def _invert_(self, encoder):
        '''Invert the encoder in order to create the decoder as a (more or less) mirror image of the encoder

        The decoder is comprised of two principal types: the 2D transpose convolution and the 2D unpooling. The 2D transpose
        convolution is followed by batch normalization and activation. Therefore as the module list of the encoder
        is iterated over in reverse, a convolution in encoder is turned into transposed convolution plus normalization
        and activation, and a maxpooling in encoder is turned into unpooling.

        Args:
            encoder (ModuleList): the encoder

        Returns:
            decoder (ModuleList): the decoder obtained by "inversion" of encoder

        '''
        modules_transpose = []
        for module in reversed(encoder):

            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels' : module.out_channels, 'out_channels' : module.in_channels,
                          'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                # module_transpose = nn.MaxUnpool2d(**kwargs)
                module_transpose = nn.Upsample(scale_factor=2, mode='nearest')
                modules_transpose += [module_transpose]

            if isinstance(module, nn.Linear):
                kwargs = {'in_features' : module.out_features, 'out_features' : module.in_features}
                module_transpose = nn.Linear(**kwargs)
                module_dropout = nn.Dropout(p=0.5, inplace=True)
                module_act =  nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_dropout, module_act]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]

        return nn.ModuleList(modules_transpose)

class VGGauthoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, B, Compression=False,train=True, save_path=None):
        x = self.encoder(x, B, Compression=Compression, train=train, save_path=save_path)
        if Compression:
          x = self.decoder.decompressed(save_path=save_path)

        x = self.decoder(x)

        return x


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']

def convrelu_mobilenet(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        self.sigmoid = nn.Sigmoid()

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, B, Compression=False, train=True, save_path=None):
        outputs = []
        shapes = []
        for i, l in enumerate(self.features):
            x = l(x)

            if (i ==2 or i ==4 or i == 9):
                x = self.sigmoid(x)
                outputs.append(x)
                shapes.append((x.shape, int(torch.flatten(x).shape[0])))

        x = self.conv(x)
        x = self.avgpool(x)
        outputs.append(x)
        shapes.append((x.shape, int(torch.flatten(x).shape[0])))

        if train:
          x = x + 1/2**B * random.uniform(-0.5, 0.5)
        if Compression:
          batch_size = x.shape[0]

          for i, out in enumerate(outputs):

            if i == 0:
              outs = torch.flatten(out).unsqueeze(0)
            else:
              out = torch.flatten(out).unsqueeze(0)
              outs = torch.cat((outs, out), 1)

          outs = torch.round(outs * 2**B)

          for_compress = torch.flatten(outs)
          for_compress = for_compress.tolist()
          frequency = collections.Counter(for_compress)

          for k in frequency.keys():
            frequency[k] = frequency[k]/len(for_compress)

          model = SimpleAdaptiveModel(frequency)
          coder = AECompressor(model)
          compressed = coder.compress(for_compress)

          if save_path != None:
            length = len(for_compress)
            torch_compressed = torch.tensor(compressed, dtype=torch.int8)
            list_to_save = [torch_compressed, coder, batch_size, length, B, shapes]

            with open(save_path, "wb") as fp:   #Pickling
              pickle.dump(list_to_save, fp)

        return outputs

class decoder_Mobilenet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer2_1x1 = convrelu_mobilenet(24, 24, 1, 0)
        self.layer4_1x1 = convrelu_mobilenet(40, 40, 1, 0)
        self.layer9_1x1 = convrelu_mobilenet(96, 96, 1, 0)
        self.out_1x1 = convrelu_mobilenet(576, 96, 1, 0)

        self.upsample7 = nn.Upsample(scale_factor=7, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')


        self.conv_up4 = convrelu_mobilenet(96 + 96, 48, 3, 1)
        self.conv_up3 = convrelu_mobilenet(48, 40, 3, 1)
        self.conv_up2 = convrelu_mobilenet(40 + 40, 24, 3, 1)
        self.conv_up1 = convrelu_mobilenet(24 + 24, 16, 3, 1)
        self.conv_up0 = convrelu_mobilenet(16, 3, 3, 1)


        self.conv_96 = nn.Sequential(convrelu_mobilenet(96, 96, 3, 1), convrelu_mobilenet(96, 96, 3, 1))
        self.conv_48 = convrelu_mobilenet(48, 48, 3, 1)
        self.conv_40 = nn.Sequential(convrelu_mobilenet(40, 40, 3, 1), convrelu_mobilenet(40, 40, 3, 1))
        self.conv_24 = convrelu_mobilenet(24, 24, 3, 1)
        self.conv_16 = nn.Sequential(convrelu_mobilenet(16, 16, 3, 1), nn.Upsample(scale_factor=2, mode='nearest'), convrelu_mobilenet(16, 16, 3, 1))

        self.conv_last = nn.Conv2d(3, 3, 1)

    def decompressed(self, save_path=None):

        if save_path != None:
            with open(save_path, 'rb') as pf:
                list_to_save = pickle.load(pf)
            pf.close()

        torch_compressed, coder, batch_size, length, B, shapes = list_to_save
        torch_compressed_list = torch_compressed.tolist()
        decompressed_list = coder.decompress(torch_compressed_list, length_encoded=length)
        decompressed_tensor = torch.FloatTensor(decompressed_list).to(device)
        decompressed_tensor = decompressed_tensor / 2 ** B

        old_index = 0
        outputs = []
        for sh in shapes:
            shape, dot = sh
            new_index = dot + old_index

            x = decompressed_tensor[old_index:new_index]
            x = torch.reshape(x, shape)
            outputs.append(x)

            old_index = new_index

        return outputs

    def forward(self, outputs):

        layer2, layer4, layer9, x  = outputs

        x = self.upsample7(x)
        x = self.out_1x1(x)

        layer9 = self.layer9_1x1(layer9)
        x = torch.cat([x, layer9], dim=1)
        x = self.conv_up4(x)

        x = self.upsample2(x)
        x = self.conv_48(x)
        x = self.conv_up3(x)

        layer4 = self.layer4_1x1(layer4)
        x = torch.cat([x, layer4], dim=1)
        x = self.conv_up2(x)

        x = self.upsample2(x)
        x = self.conv_24(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up1(x)

        x = self.upsample2(x)
        x = self.conv_16(x)

        x = self.upsample2(x)
        x = self.conv_up0(x)

        out = self.conv_last(x)

        return out

class Mobilenet_authoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, B, Compression=False,train=True, save_path=None):
        outputs = self.encoder(x, B, Compression=Compression, train=train, save_path=save_path)

        if Compression:
          outputs = self.decoder.decompressed(save_path=save_path)

        out = self.decoder(outputs)

        return out