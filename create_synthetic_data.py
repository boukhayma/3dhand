import sys
sys.path.insert(0, "mano/")

import pickle
import random
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from webuser.smpl_handpca_wrapper_HAND_only import load_model
import cv2

import scipy.misc as misc



# Please add fuction "change_col()" to ColoredRenderer class in file renderer.py in your opendr
#class ColoredRenderer(BaseRenderer):  
#    def change_col(self, color):
#        self.vc = color



# total number of bg images
bg_number = 3
m = load_model('mano/models/MANO_RIGHT.pkl', ncomps=6, flat_hand_mean=False)

# load all colors 
colors = []
for i in xrange(0,27):
    f = open('data/meshes_colored/%d.obj'%i)
    cont = f.readlines() 
    f.close() 

    col=[]
    for x in cont:
        a = x[:len(x)-1].split(" ")
        if (a[0] == 'v'):
            col.append(np.array([float(a[4]),float(a[5]),float(a[6])])) 
    col = np.expand_dims(np.vstack(col),0)
    colors.append(col)
    
colors = np.vstack(colors)



# generate synthetic images
joints = []
gtruth = []

for i in xrange(0,3):

    m.betas[:] =  np.array([random.uniform(-1.,1.) for _ in xrange(10)]) * .03
    m.pose[:] = np.array([random.uniform(-1.,1.) for _ in xrange(9)]) * 2.
    m.pose[:3] = [np.pi, 0., 0.]

    angle = random.uniform(-np.pi,np.pi)
    axis =  np.array([random.uniform(-1.,1.) for _ in xrange(3)]) 
    axis[random.randint(0,2)] = 1.  
    axis /= np.linalg.norm(axis) 
    rot = angle * axis   
    
    R = cv2.Rodrigues(rot)[0] 
 
    w, h = (320, 320)
    rn = ColoredRenderer()

    mesh = m.r
    joint = m.J_transformed.r

    mesh = np.transpose(np.matmul(R,np.transpose(mesh)))
    joint = np.transpose(np.matmul(R,np.transpose(joint)))
    
    umax = np.max(mesh[:,0])
    umin = np.min(mesh[:,0])         
    vmax = np.max(mesh[:,1])
    vmin = np.min(mesh[:,1])    

    c = random.uniform(2.0, 2.4) * np.max([umax - umin, vmax - vmin]) 
    ss = 320.0/c
      
    mesh = np.array([[ss,ss,1],]*778)*mesh
    joint = np.array([[ss,ss,1],]*16)*joint

    umax = np.max(mesh[:,0])
    umin = np.min(mesh[:,0])         
    vmax = np.max(mesh[:,1])
    vmin = np.min(mesh[:,1])  
    
    tumax = 319-umax
    tumin = umin
    tvmax = 319-vmax
    tvmin = vmin
    
    tu = random.uniform(-tumin,tumax) 
    tv = random.uniform(-tvmin,tvmax) 
    
    mesh = mesh + np.array([[tu,tv,0],]*778)
    joint = joint + np.array([[tu,tv,0],]*16)

    umax = np.max(mesh[:,0])
    umin = np.min(mesh[:,0])         
    vmax = np.max(mesh[:,1])
    vmin = np.min(mesh[:,1])

    if ((umin<0.) or (vmin<0.) or (umax>320.) or (vmax>320.)):
        print('mesh outside')
        
    mesh[:,2] = 10.0 + (mesh[:,2]-np.mean(mesh[:,2]))
    mesh[:,:2] = mesh[:,:2] * np.expand_dims(mesh[:,2],1)

    rn.camera = ProjectPoints(v=mesh, rt=np.zeros(3), t=np.array([0, 0, 0]), f=np.array([1,1]), c=np.array([0,0]), k=np.zeros(5))

    rn.frustum = {'near': 1., 'far': 20., 'width': w, 'height': h}
    rn.set(v=mesh, f=m.f, bgcolor=np.zeros(3))  
    rn.vc = LambertianPointLight(f=m.f, v=mesh, num_verts=len(m),light_pos=np.array([0,0,0]),vc=np.ones_like(m)*.9,light_color=np.array([1., 1., 1.]))

    mod = i % bg_number
    bg = misc.imread('data/backgrounds/%d.png'%(mod))

    rn.change_col(np.ones((778,3)))

    mask = rn.r.copy()   
    mask = mask[:,:,0].astype(np.uint8)

    rn.change_col(colors[random.randint(0,26)])
    
    hand = rn.r.copy()*255.    
    image = (1-np.expand_dims(mask,2)) * bg + np.expand_dims(mask,2) * hand  

    # image
    misc.imsave('data/out/%d.png'%i, image)
    # segmentation    
    misc.imsave('data/out/mask_%d.png'%i, mask*255)

    # joint locations
    joints.append(joint[:,:2].reshape((32)))
    # hand and view gt parameters
    gtruth.append(np.concatenate([np.array([1.,ss,tu,tv]),rot,m.pose[3:],m.betas],0))


with open('data/out/labels.pickle','wb') as fo:
    pickle.dump(joints,fo,protocol=pickle.HIGHEST_PROTOCOL) 

with open('data/out/gt.pickle','wb') as fo:
    pickle.dump(gtruth,fo,protocol=pickle.HIGHEST_PROTOCOL) 








