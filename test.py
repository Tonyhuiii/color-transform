#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model import Encoder
from model import Decoder
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--encoder', type=str, default='checkpoint/527wt/encoder_80.pth', help='encoder checkpoint file')
parser.add_argument('--decoder', type=str, default='checkpoint/527wt/decoder_80.pth', help='decoder checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
encoder = Encoder(opt.input_nc)
decoder = Decoder()

if opt.cuda:
    encoder.cuda()
    decoder.cuda()

# Load state dicts
encoder.load_state_dict(torch.load(opt.encoder))
decoder.load_state_dict(torch.load(opt.decoder))

# Set model's test mode
encoder.eval()
decoder.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######


# Create output dirs if they don't exist
fakeB_path='output/'+opt.dataroot.split('/')[-1]+'/fake_B'

if not os.path.exists(fakeB_path):
    os.makedirs(fakeB_path)


for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    filename= batch['filename'][0]
    real_A = -real_A

    # Generate output
    fake_B = 0.5*(decoder(encoder(real_A)) + 1.0)


    # Save image files
    save_image(fake_B, fakeB_path+'/{}.jpg'.format(filename),padding=0)

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
