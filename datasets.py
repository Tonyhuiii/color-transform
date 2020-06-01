import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode=mode
        
        # self.transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=1),
        #                                       transforms.ToTensor()])
        
        # self.transform2 = transforms.Compose([transforms.ToTensor(),
        #         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.filename=[x.split('/')[-1][:-4] for x in  self.files_A]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            
        if (self.mode=='train'):
            return {'A': item_A, 'B': item_B}
        else:
            return  {'A': item_A, 'B': item_B, 'filename': self.filename[index]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))