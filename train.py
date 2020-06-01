#!/usr/bin/python3

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import os

from model import Encoder
from model import Decoder
from model import VGGLoss
from model import MultiscaleDiscriminator
from model import transformer_block
from model import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
   
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
encoder = Encoder(opt.input_nc)
decoder = Decoder()
# netD = Discriminator(opt.input_nc)
netD = MultiscaleDiscriminator(opt.input_nc, opt.ndf, opt.n_layers_D, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, num_D=1, getIntermFeat=False)   
# transformer=transformer_block()


if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    netD.cuda()
    # transformer.cuda()

encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
netD.apply(weights_init_normal)


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()
criterion_feat = torch.nn.MSELoss()
criterion_VGG= VGGLoss()

# Optimizers & LR schedulers
optimizer_encoder = torch.optim.Adam(encoder.parameters(),lr=opt.lr, betas=(0.5, 0.999))
optimizer_decoder = torch.optim.Adam(decoder.parameters(),lr=opt.lr, betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_t = torch.optim.Adam(transformer.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_encoder = torch.optim.lr_scheduler.LambdaLR(optimizer_encoder, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_decoder = torch.optim.lr_scheduler.LambdaLR(optimizer_decoder, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# lr_scheduler_t = torch.optim.lr_scheduler.LambdaLR(optimizer_t, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
# input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

# Create checkpoint dirs if they don't exist
checkpoint_path='checkpoint/'+opt.dataroot.split('/')[-1]
# print(checkpoint_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

###
def GANloss(predict, target_is_real):
    loss=0
    # print(len(predict[0]))
    for pred in predict[0]:
        if target_is_real:
            target = Variable(Tensor(pred.size()).fill_(1.0), requires_grad=False)
        else:
            
            target = Variable(Tensor(pred.size()).fill_(0.0), requires_grad=False)
        loss += criterion_GAN(pred, target) 
    return loss

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        A = batch['A'].cuda()
        real_A= -A
        
        real_B = batch['B'].cuda()
        
        # real_A= real_A[:,2,:,:]
        # real_A= real_A.view(real_B.shape[0],-1,real_B.shape[2],real_B.shape[3])
        # real_A= real_A.expand(-1,3,-1,-1)


        real_A_features=encoder(real_A)
        fake_B=decoder(real_A_features)
        fake_B_features=encoder(fake_B.detach())

        # fake=(fake_B+1)*0.5
        # R = fake[:,0,:,:]
        # G = fake[:,1,:,:]
        # B = fake[:,2,:,:]
        # gray_B=0.299*R+0.587*G+0.114*B
        # gray_B= gray_B.view(fake_B.shape[0],-1,fake_B.shape[2],fake_B.shape[3])
        # gray_B=gray_B.expand(-1,3,-1,-1)
        # loss_gray=criterion_l1(gray_A,gray_B)*5

        ###### Generators  ######
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        
        # GAN loss
        pred_fake = netD.forward(fake_B)
        loss_G = GANloss(pred_fake, True)


        # Features loss
        loss_feature=criterion_l1(real_A_features, fake_B_features)*10
        
        # Image loss
        # A=transformer_block(real_A)
        # B=transformer_block(fake_B)
        # loss_img = criterion_GAN(A,B)
        
        # Identity loss
        idt_B = decoder(encoder(real_B))
        loss_idt = criterion_l1(idt_B,real_B)*5
        
        # GAN feature matching loss
        # loss_feat = 0
        # for j in range(len(pred_fake[0])-1):
        #     loss_feat += criterion_feat(pred_fake[0][j], pred_real[0][j].detach())*1
                        
        # VGG feature matching loss
        loss_VGG = criterion_VGG(fake_B,real_A)*10
        
           
        ##ssimloss
        ssim_module = SSIM(data_range=1, size_average=True, channel=3)
        X1 = (real_A + 1)*0.5   # [-1, 1] => [0, 1]
        Y1 = (fake_B + 1)*0.5  
        ssim_A = (1 - ssim_module(X1, Y1))


        # Total loss
        loss = loss_G + loss_feature + loss_idt + ssim_A 
        
        loss.backward()
        
        optimizer_encoder.step()
        optimizer_decoder.step()
        ###################################

        
        ###### Discriminator ######
        optimizer_D.zero_grad()

        # Real loss
        pred_real = netD.forward(real_B)

        loss_D_real = GANloss(pred_real, True)

        # print(pred_real.size()[0])
        # target_real = Variable(Tensor(pred_real.size()[0]).fill_(1.0), requires_grad=False)
        # print(target_real.shape)
        # target_fake = Variable(Tensor(pred_real.size()[0]).fill_(0.0), requires_grad=False)

        # loss_D_real = criterion_GAN(pred_real, target_real)
 
        # Fake loss
        fake_B1= fake_B
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD.forward(fake_B.detach())
        loss_D_fake = GANloss(pred_fake, False)        

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss': loss,  'loss_G': loss_G, 'loss_feature': loss_feature, 'loss_idt': loss_idt,
                    'loss_VGG': loss_VGG,
                    'ssim_A': ssim_A,                    
                    'loss_D': loss_D}, 
                    images={'A': A,'real_A': real_A,'real_B': real_B, 'fake_B': fake_B1,'idt_B':idt_B})

    # Update learning rates
    lr_scheduler_D.step()
    # lr_scheduler_t.step()
    lr_scheduler_encoder.step()
    lr_scheduler_decoder.step()

    # Save models checkpoints
    if ((epoch+1)%10==0):
        torch.save(encoder.state_dict(), checkpoint_path+'/encoder_{}.pth'.format(epoch+1))
        torch.save(decoder.state_dict(), checkpoint_path+'/decoder_{}.pth'.format(epoch+1))
        torch.save(netD.state_dict(), checkpoint_path+'/netD_{}.pth'.format(epoch+1))

