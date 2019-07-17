import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2 as cv

for ii in xrange(3):             
    
    with open('data/original/'+str(ii)+'.json', 'r') as fid:
        dat = json.load(fid)
    pts = np.array(dat['hand_pts'])
    image = misc.imread('data/original/'+str(ii) +'.jpg')

    vsz, usz = image.shape[:2]
    minsz = min(usz,vsz)
    maxsz = max(usz,vsz)

    kp_visible = (pts[:, 2] == 1)

    uvis = pts[kp_visible,0]
    vvis = pts[kp_visible,1]    
    
    umin = min(uvis)
    vmin = min(vvis)
    umax = max(uvis)
    vmax = max(vvis) 

    B = round(2.2 * max([umax-umin, vmax-vmin]))    

    us = 0
    ue = usz-1 

    vs = 0
    ve = vsz-1 

    umid = umin + (umax-umin)/2 
    vmid = vmin + (vmax-vmin)/2 
     
    if (B < minsz-1): 
                        
        us = round(max(0, umid - B/2))
        ue = us + B

        if (ue>usz-1):
            d = ue - (usz-1)
            ue = ue - d
            us = us - d

        vs = round(max(0, vmid - B/2))
        ve = vs + B

        if (ve>vsz-1):
            d = ve - (vsz-1)
            ve = ve - d
            vs = vs - d    
        
    if (B>=minsz-1):    
        
        B = minsz-1
        if usz == minsz:           
            vs = round(max(0, vmid - B/2))
            ve = vs + B

            if (ve>vsz-1):
                d = ve - (vsz-1)
                ve = ve - d
                vs = vs - d    

        if vsz == minsz:
            us = round(max(0, umid - B/2))
            ue = us + B

            if (ue>usz-1):
                d = ue - (usz-1)
                ue = ue - d
                us = us - d        

    us = int(us)
    vs = int(vs)
    ue = int(ue)
    ve = int(ve)        
    
    uvis  = (uvis - us) * (319.0/(ue-us)) 	
    vvis  = (vvis - vs) * (319.0/(ve-vs))     

    img = misc.imresize(image[vs:ve+1,us:ue+1,:], (320, 320), interp='bilinear')
    misc.imsave('data/out/'+str(ii)+'.png',img) 



