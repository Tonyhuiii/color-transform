#!/usr/bin/python3

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Encoder
from model import Decoder
from model import transformer_block
from model import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

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
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
encoder = Encoder(opt.input_nc)
decoder = Decoder()
netD = Discriminator(opt.input_nc)
# transformer=transformer_block()


if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    netD.cuda()
    # transformer.cuda()

encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
netD.apply(weights_init_normal)
# transformer.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()

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

# fake_B_buffer = ReplayBuffer()

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

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        # real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))

        real_A_features=encoder(real_A)
        fake_B=decoder(real_A_features)
        fake_B_features=encoder(fake_B.detach())
#        print(real_A.shape)
#        print(real_A_features.shape)
#        print(fake_B.shape)
#        print(fake_B_features.shape)
        
        
        
        ###### Discriminator ######
        optimizer_D.zero_grad()

        # Real loss
        pred_real = netD(real_B)
        # print(pred_real.size()[0])
        target_real = Variable(Tensor(pred_real.size()[0]).fill_(1.0), requires_grad=False)
        # print(target_real.shape)
        target_fake = Variable(Tensor(pred_real.size()[0]).fill_(0.0), requires_grad=False)

        loss_D_real = criterion_GAN(pred_real, target_real)
 
        # Fake loss
        pred_fake1 = netD(fake_B.detach())
        loss_D_fake1 = criterion_GAN(pred_fake1, target_fake)

        pred_fake2 = netD(real_A)
        loss_D_fake2 = criterion_GAN(pred_fake2, target_fake)
        
        # Total loss
        loss_D = loss_D_real + loss_D_fake1 + loss_D_fake2
        loss_D.backward()

        optimizer_D.step()
        ###################################

        ###### Generators  ######
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        
        # GAN loss
        pred_fake = netD(fake_B)
        loss_G = criterion_GAN(pred_fake, target_real)
        
        # loss_G.backward(retain_graph=True)
    

        # Features loss
        loss_feature=criterion_l1(real_A_features, fake_B_features)

        # loss_feature.backward()
        # optimizer_encoder.step()
        
        # Image loss
        # A=transformer_block(real_A)
        # B=transformer_block(fake_B)
        # loss_img = criterion_GAN(A,B)
        
        # Identity loss
        idt_A = decoder(encoder(real_B))
        loss_idt = criterion_l1(idt_A,real_B)

        # Total loss
        loss = loss_G+loss_feature*10+loss_idt*10
        
        loss.backward()
        
        optimizer_encoder.step()
        optimizer_decoder.step()
        ###################################



        # Progress report (http://localhost:8097)
        logger.log({'loss': loss,  'loss_G': loss_G, 'loss_feature': loss_feature, 'loss_idt': loss_idt,'loss_D': loss_D}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B,'idt_A':idt_A})

    # Update learning rates
    lr_scheduler_D.step()
    # lr_scheduler_t.step()
    lr_scheduler_encoder.step()
    lr_scheduler_decoder.step()

    # Save models checkpoints
    if ((epoch+1)%10==0):
        torch.save(encoder.state_dict(), 'output/encoder_{}.pth'.format(epoch+1))
        torch.save(decoder.state_dict(), 'output/decoder_{}.pth'.format(epoch+1))
        torch.save(netD.state_dict(), 'output/netD_{}.pth'.format(epoch+1))

