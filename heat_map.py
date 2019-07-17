import numpy as np
import os
import scipy.misc as misc
import PyOpenPose as OP
import cv2

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]
ll = [[0,5,6],[7,8,9],[10,11,12],[17,18,19],[20,13,14],[15,16,1],[2,3,4]]
download_heatmaps = True
with_hands = True
with_face = False
handBB = [0, 0, 320, 320]
op = OP.OpenPose((656, 368), (320, 320), (320,320), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

for i in xrange(3):
    
    rgb = cv2.imread('/media/hdd-0/hands/code/final/data/cropped/%d.png' %i)
    rgb = cv2.flip(rgb,1)

    vec = np.array(handBB + [0, 0, 0, 0], dtype=np.int32).reshape((1, 8))       
    op.detectHands(rgb, vec)  
    left_hands = op.getHandHeatmaps()[0]
    
    for j,l in enumerate(ll):

        hm0 = left_hands[0,l[0],:, ::-1]
        hm1 = left_hands[0,l[1],:, ::-1]
        hm2 = left_hands[0,l[2],:, ::-1]

        hm0 = np.expand_dims(np.fliplr(hm0*255).astype(np.uint8),axis=2)
        hm1 = np.expand_dims(np.fliplr(hm1*255).astype(np.uint8),axis=2)
        hm2 = np.expand_dims(np.fliplr(hm2*255).astype(np.uint8),axis=2)
 
        hm = np.concatenate((hm0,hm1,hm2),axis=2)     
 
        misc.imsave('/media/hdd-0/hands/code/final/data/out/%d_%d.png' %(i,j),hm)    



