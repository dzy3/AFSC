import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F


### Dilated Convolution with Dense Connection module ###
class denseBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize=3, inception=False, dilateScale=1, activ='ReLU'):
        super(denseBlockLayer, self).__init__()
        self.useInception = inception

        if (self.useInception):
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
            self.conv3 = nn.Conv2d(in_channels, out_channels, 7, padding=3)
            if (activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            self.conv4 = nn.Conv2d(out_channels * 3, out_channels, 1, padding=0)
        else:
            pad = int(dilateScale * (kernelSize - 1) / 2)

            self.conv = nn.Conv2d(in_channels, out_channels, kernelSize, padding=pad, dilation=dilateScale)
            if (activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()

    def forward(self, x):
        if (self.useInception):
            y2 = x
            y3_1 = self.conv1(y2)
            y3_2 = self.conv1(y2)
            y3_3 = self.conv1(y2)
            y4 = torch.cat((y3_1, y3_2, y3_3), 1)
            y4 = self.relu(y4)
            y5 = self.conv4(y4)
            y_ = self.relu(y5)
        else:
            y2 = self.conv(x)
            y_ = self.relu(y2)

        return y_


class denseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize=3, growthRate=16, layer=3, inceptionLayer=False,
                 dilationLayer=True, activ='ReLU'):
        super(denseBlock, self).__init__()
        dilate = 1
        if (dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer(in_channels + growthRate * i, growthRate, kernelSize, inceptionLayer, dilate,
                                        activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
            print(dilate)
        self.layerList = nn.ModuleList(templayerList)
        self.outputLayer = denseBlockLayer(in_channels + growthRate * layer, out_channels, kernelSize, inceptionLayer,
                                           1,
                                           activ)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.outputLayer(x)
        y = self.bn(y)

        return y
################################################


### Adaptive Fourier Space Compression module ###
class Reconstruction_Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(Reconstruction_Network, self).__init__()
        self.encoder = Encoder_Reconstruction_Network(in_channels, base_width)  # The Encoder part of the reconstruction network
        self.decoder = Decoder_Reconstruction_Network(base_width, out_channels=out_channels) # The Decoder part of the reconstruction network

        # Initialization of AFSCm-related variables
        mult1 = torch.rand(1, 256, 256)
        while mult1.min == 0:
            mult1 = torch.rand(1, 256, 256)
        self.slope1_1 = 10.0
        mult1 = - torch.log(1. / mult1 - 1.) / self.slope1_1
        self.mult1 = torch.nn.Parameter(mult1, requires_grad=True)
        self.slope1_2 = 12.0

        mult2 = torch.rand(1, 256, 256)
        while mult2.min == 0:
            mult2 = torch.rand(1, 256, 256)
        self.slope2_1 = 10.0
        mult2 = - torch.log(1. / mult2 - 1.) / self.slope2_1
        self.mult2 = torch.nn.Parameter(mult2, requires_grad=True)
        self.slope2_2 = 12.0

        mult3 = torch.rand(1, 256, 256)
        while mult3.min == 0:
            mult3 = torch.rand(1, 256, 256)
        self.slope3_1 = 10.0
        mult3 = - torch.log(1. / mult3 - 1.) / self.slope3_1
        self.mult3 = torch.nn.Parameter(mult3, requires_grad=True)
        self.slope3_2 = 12.0

        self.frequency_mask1 = torch.zeros(mult1.shape).cuda()
        self.frequency_mask2 = torch.zeros(mult2.shape).cuda()
        self.frequency_mask3 = torch.zeros(mult3.shape).cuda()

    def forward(self, x, len_dataloader, i_batch, epoch, model_name, sparsity1=0.5, sparsity2=0.5, sparsity3=0.5):
        # len_dataloader: Obtained via ->  len(dataloader)
        # i_batch: Obtained via for  ->  i_batch, x in enumerate(dataloader):
        # epoch: Indicates the count of the current epoch.
        # model_name: Name of the model.
        # sparsity(1,2,3): The sparsity corresponding to the RGB channels.

        fft_im_ = torch.fft.fft2(x, dim=(-2, -1))

        Fourier_tensor = torch.cat([torch.real(fft_im_), torch.imag(fft_im_)], dim=1)

        if epoch < 400:  # When the model is at > 399th epoch, the mask will no longer be optimized with the network.

            intermediate_tensor1 = torch.sigmoid(self.slope1_1 * self.mult1)   # (1,256,256)
            mu1 = (0 * intermediate_tensor1) + torch.rand(1, 256, 256).cuda()
            Fourier_tensor_mask1 = torch.sigmoid(self.slope1_2 * (intermediate_tensor1 - mu1))


            intermediate_tensor2 = torch.sigmoid(self.slope2_1 * self.mult2)   # (1,256,256)
            mu2 = (0 * intermediate_tensor2) + torch.rand(1, 256, 256).cuda()
            Fourier_tensor_mask2 = torch.sigmoid(self.slope2_2 * (intermediate_tensor2 - mu2))

            intermediate_tensor3 = torch.sigmoid(self.slope3_1 * self.mult3)   # (1,256,256)
            mu3 = (0 * intermediate_tensor3) + torch.rand(1, 256, 256).cuda()
            Fourier_tensor_mask3 = torch.sigmoid(self.slope3_2 * (intermediate_tensor3 - mu3))

            if epoch == 399:  # At the 399th epoch, the mask will be binarised and saved.
                with torch.no_grad():
                    if i_batch == 0:
                        frequency_mask_flod_path = "./save_masks/" + model_name + '/'
                        if not os.path.exists(frequency_mask_flod_path):
                            os.makedirs(frequency_mask_flod_path)

                    self.frequency_mask1 += Fourier_tensor_mask1.detach()
                    self.frequency_mask2 += Fourier_tensor_mask2.detach()
                    self.frequency_mask3 += Fourier_tensor_mask3.detach()

                    if i_batch == len_dataloader - 1:
                        frequency_mask_flod_path = "./save_masks/" + model_name + '/'
                        nums = (Fourier_tensor_mask1.shape[-1]) * (Fourier_tensor_mask1.shape[-2]) * (
                        Fourier_tensor_mask1.shape[-3])

                        pointer1 = torch.sort(self.frequency_mask1.flatten()).values[int(nums - nums * sparsity1 - 1)]
                        frequency_mask_1 = torch.where(self.frequency_mask1 > pointer1, 1.0, 0.0)
                        frequency_mask_path1 = frequency_mask_flod_path + 'frequency_mask1'
                        np.save(frequency_mask_path1, frequency_mask_1.detach().cpu().numpy())

                        pointer2 = torch.sort(self.frequency_mask2.flatten()).values[int(nums - nums * sparsity2 - 1)]
                        frequency_mask_2 = torch.where(self.frequency_mask2 > pointer2, 1.0, 0.0)
                        frequency_mask_path2 = frequency_mask_flod_path + 'frequency_mask2'
                        np.save(frequency_mask_path2, frequency_mask_2.detach().cpu().numpy())

                        pointer3 = torch.sort(self.frequency_mask3.flatten()).values[int(nums - nums * sparsity3 - 1)]
                        frequency_mask_3 = torch.where(self.frequency_mask3 > pointer3, 1.0, 0.0)
                        frequency_mask_path3 = frequency_mask_flod_path + 'frequency_mask3'
                        np.save(frequency_mask_path3, frequency_mask_3.detach().cpu().numpy())

        else:
            frequency_mask1 = np.load('./save_masks/' + model_name + '/' + 'frequency_mask1.npy')
            frequency_mask2 = np.load('./save_masks/' + model_name + '/' + 'frequency_mask2.npy')
            frequency_mask3 = np.load('./save_masks/' + model_name + '/' + 'frequency_mask3.npy')
            Fourier_tensor_mask1 = torch.from_numpy(frequency_mask1).cuda()
            Fourier_tensor_mask2 = torch.from_numpy(frequency_mask2).cuda()
            Fourier_tensor_mask3 = torch.from_numpy(frequency_mask3).cuda()

        # Downsampling of the image Fourier space by the masks

        Fourier_space_r1 = torch.multiply(Fourier_tensor[:, 0:1, :, :], Fourier_tensor_mask1.unsqueeze(0))
        Fourier_space_r2 = torch.multiply(Fourier_tensor[:, 1:2, :, :], Fourier_tensor_mask2.unsqueeze(0))
        Fourier_space_r3 = torch.multiply(Fourier_tensor[:, 2:3, :, :], Fourier_tensor_mask3.unsqueeze(0))

        Fourier_space_i1 = torch.multiply(Fourier_tensor[:, 3:4, :, :], Fourier_tensor_mask1.unsqueeze(0))
        Fourier_space_i2 = torch.multiply(Fourier_tensor[:, 4:5, :, :], Fourier_tensor_mask2.unsqueeze(0))
        Fourier_space_i3 = torch.multiply(Fourier_tensor[:, 5:6, :, :], Fourier_tensor_mask3.unsqueeze(0))

        x_real = torch.cat([Fourier_space_r1, Fourier_space_r2, Fourier_space_r3], dim=1)
        x_imag = torch.cat([Fourier_space_i1, Fourier_space_i2, Fourier_space_i3], dim=1)

        fft_src_ = torch.complex(x_real, x_imag)

        # Convert to image space
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1))
        src_in_trg = torch.real(src_in_trg)
        x_ = src_in_trg.float()

        # Inputting the compressed image to the network
        embedding = self.encoder(x_)
        output = self.decoder(embedding)
        return output, Fourier_tensor_mask1, Fourier_tensor_mask2, Fourier_tensor_mask3



# Example of how our blocks should be uesed
# x_rec, Fourier_tensor_mask1, Fourier_tensor_mask2, Fourier_tensor_mask3 = model(x,len(dataloader),
#                                                                                   i_batch,
#                                                                                   epoch, run_name,
#                                                                                   args.sparsity1,
#                                                                                   args.sparsity2,
#                                                                                   args.sparsity3)
# if epoch < 400:
#     regularization_loss = torch.sum(torch.abs(Fourier_tensor_mask1)) + torch.sum(
#         torch.abs(Fourier_tensor_mask2)) + torch.sum(torch.abs(Fourier_tensor_mask3))
#     loss = loss_reconstruction(x_rec,x) + 0.000001 * regularization_loss
# else:
#     loss = loss_reconstruction(x_rec,x)