import os
from PIL import Image
import torch
from torch.utils import data

class HandTestSet(data.Dataset):
    def __init__(self, root, img_transform=None):
        self.data_dir = root
        self.img_transform = img_transform
                                
    def __len__(self):
        return 3
        
    def __getitem__(self, index):
        imgs = [self.img_transform(Image.open(os.path.join(self.data_dir, '%d.png' % index)).convert('RGB') )]        
        for i in xrange(7):
            imgs.append(self.img_transform(Image.open(os.path.join(self.data_dir, '%d_%d.png' %(index,i))).convert('RGB')))
        imgs = torch.cat(imgs,dim=0)
        
        return imgs








