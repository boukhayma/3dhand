import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np


#-------------------
# Mano in Pytorch
#-------------------

bases_num = 10 
pose_num = 6
mesh_num = 778
keypoints_num = 16
 
dd = pickle.load(open('mano/models/MANO_RIGHT.pkl', 'rb'))
kintree_table = dd['kintree_table']
id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])} 
parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}  

mesh_mu = Variable(torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).cuda()) # zero mean
mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).cuda())
posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).cuda())
J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).cuda())
weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).cuda())
hands_components = Variable(torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:pose_num]), 0).astype(np.float32)).cuda())
hands_mean       = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).cuda())
root_rot = Variable(torch.FloatTensor([np.pi,0.,0.]).unsqueeze(0).cuda())

def rodrigues(r):       
    theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))  

    def S(n_):   
        ns = torch.split(n_, 1, 1)     
        Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)      
        return Sn_    

    n = r/(theta.view(-1, 1))   
    Sn = S(n) 

    #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)
    
    I3 = Variable(torch.eye(3).unsqueeze(0).cuda())

    R = I3 + torch.sin(theta).view(-1, 1, 1)*Sn\
        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    Sr = S(r)
    theta2 = theta**2     
    R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
        + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)
    
    idx = np.argwhere((theta<1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx,:,:] = R2[idx,:,:]

    return R,Sn

def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3                                                    
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))    
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix

def rot_pose_beta_to_mesh(rots, poses, betas):

    batch_size = rots.size(0)   

    poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,keypoints_num-1,3)
    #poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)   
    poses = torch.cat((root_rot.repeat(batch_size,1).view(batch_size,1,3),poses),1)

    v_shaped =  (torch.matmul(betas.unsqueeze(1), 
                mesh_pca.repeat(batch_size,1,1,1).permute(0,3,1,2).contiguous().view(batch_size,bases_num,-1)).squeeze(1)    
                + mesh_mu.repeat(batch_size,1,1).view(batch_size, -1)).view(batch_size, mesh_num, 3)      
    
    pose_weights = get_poseweights(poses, batch_size)    

    v_posed = v_shaped + torch.matmul(posedirs.repeat(batch_size,1,1,1),
              (pose_weights.view(batch_size,1,(keypoints_num - 1)*9,1)).repeat(1,mesh_num,1,1)).squeeze(3)

    J_posed = torch.matmul(v_shaped.permute(0,2,1),J_regressor.repeat(batch_size,1,1).permute(0,2,1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]
         
    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)


    angle_matrix =[]
    for i in range(keypoints_num):
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    #with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x:\
        torch.cat((x,   Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1).cuda())  ),1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size,4,3).cuda()),x),2) 

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size,3,1)),2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                         (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size,3,1)),2)) 
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []
         
    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size,1).cuda()) ),1)).view(batch_size,4,1)
        results2.append((results[i]-pack(torch.matmul(results[i], vec))).unsqueeze(0))    

    results = torch.cat(results2, 0)
    
    T = torch.matmul(results.permute(1,2,3,0), weights.repeat(batch_size,1,1).permute(0,2,1).unsqueeze(1).repeat(1,4,1,1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size,mesh_num,1).cuda()) ), 2)  
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num)\
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num)\
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num)\
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)
   
    #v = v.permute(0,2,1)[:,:,:3] 
    Rots = rodrigues(rots)[0]

    Jtr = []

    for j_id in range(len(results_global)):
        Jtr.append(results_global[j_id][:,:3,3:4])

    # Add finger tips from mesh to joint list    
    Jtr.insert(4,v[:,:3,333].unsqueeze(2))
    Jtr.insert(8,v[:,:3,444].unsqueeze(2))
    Jtr.insert(12,v[:,:3,672].unsqueeze(2))
    Jtr.insert(16,v[:,:3,555].unsqueeze(2))
    Jtr.insert(20,v[:,:3,745].unsqueeze(2))        
     
    Jtr = torch.cat(Jtr, 2) #.permute(0,2,1)
           
    v = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)
    
    return torch.cat((Jtr,v), 1)





#-------------------
# Resnet + Mano
#-------------------

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet_Mano(nn.Module):

    def __init__(self, block, layers, input_option, num_classes=1000):

        self.input_option = input_option
        self.inplanes = 64
        super(ResNet_Mano, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)       
        #if (self.input_option):        
        self.conv11 = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)                        
        self.mean = Variable(torch.FloatTensor([545.,128.,128.,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]).cuda())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
       
        if (self.input_option):       
            x = self.conv11(x)
        else:
            x = self.conv1(x[:,0:3])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 

        xs = self.fc(x)
        xs = xs + self.mean  

        scale = xs[:,0]
        trans = xs[:,1:3] 
        rot = xs[:,3:6]    
        theta = xs[:,6:12]
        beta = xs[:,12:] 

        x3d = rot_pose_beta_to_mesh(rot,theta,beta)
        
        x = trans.unsqueeze(1) + scale.unsqueeze(1).unsqueeze(2) * x3d[:,:,:2] 
        x = x.view(x.size(0),-1)      
              
        #x3d = scale.unsqueeze(1).unsqueeze(2) * x3d
        #x3d[:,:,:2]  = trans.unsqueeze(1) + x3d[:,:,:2] 
        
        return x, x3d

def resnet34_Mano(pretrained=False,input_option=1, **kwargs):
    
    model = ResNet_Mano(BasicBlock, [3, 4, 6, 3], input_option, **kwargs)    
    model.fc = nn.Linear(512 * 1, 22)

    return model


