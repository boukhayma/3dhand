from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from model import resnet34_Mano
from datasets import HandTestSet
from transform import Scale
from torchvision.transforms import ToTensor, Compose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1 use image and joint heat maps as input
# 0 use image only as input 
input_option = 1

img_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor()])

template = open('data/template.obj')
content = template.readlines()
template.close()

testloader = data.DataLoader(HandTestSet('data/cropped', img_transform=img_transform),
                            num_workers=0,batch_size=1, shuffle=False, pin_memory=False)

model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option))    
model.load_state_dict(torch.load('data/model-' + str(input_option) + '.pth'))
model.eval()

for i, data in enumerate(testloader, 0):
    images = data
    images = Variable(images.cuda())
    out1, out2 = model(images)    
    imgs = images[0].data        

    # Display 2D joints    
    u = np.zeros(21)   
    v = np.zeros(21)   
    for ii in xrange(21): 
        u[ii] = out1[0,2*ii]
        v[ii] = out1[0,2*ii+1]                           
    plt.plot(u, v, 'ro', markersize=5)      
    fig = plt.figure(1)
    plt.imshow(imgs[:3,:,:].permute(1,2,0))
    plt.show()

    # Save 3D mesh
    file1 = open('data/out/'+str(i)+'.obj','w')   
    for j in xrange(778):
        file1.write("v %f %f %f\n"%(out2[0,21+j,0],-out2[0,21+j,1],-out2[0,21+j,2]))
    for j,x in enumerate(content):  
        a = x[:len(x)-1].split(" ")
        if (a[0] == 'f'):
            file1.write(x)  
    file1.close()   
   



