import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

def inside_polygon(x, y, points):
    n = len(points)

    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

fi = open('data/cropped/labels.pickle', 'rb')
anno = pickle.load(fi)
fi.close()

for ii in xrange(3):     

    img = misc.imread('data/cropped/'+str(ii) +'.png')
    mask = np.zeros((320,320), np.uint8)
     
    # Draw skeleton

    for e in edges:
        p1u = int(anno[ii][2*e[0]])
        p1v = int(anno[ii][2*e[0]+1]) 
        p2u = int(anno[ii][2*e[1]])
        p2v = int(anno[ii][2*e[1]+1])    

        cv.line(mask, (p1u,p1v), (p2u,p2v), 3, 70)
    
    for e in edges:
        p1u = int(anno[ii][2*e[0]])
        p1v = int(anno[ii][2*e[0]+1]) 
        p2u = int(anno[ii][2*e[1]])
        p2v = int(anno[ii][2*e[1]+1])    

        cv.line(mask, (p1u,p1v), (p2u,p2v), 1, 1)
       
    poly_list = [[0,17,18],[0,17,1],[0,1,5],[0,5,13],[0,13,9]]
    polys = []             

    # Draw triangles  
  
    for i in poly_list:
        poly = []

        for ind in i:
            pu = int(anno[ii][2*ind])
            pv = int(anno[ii][2*ind+1])             
            poly.append((pv,pu))
            polys.append(poly)    

    for u in xrange(0,320):
        for v in xrange(0,320):
            for j in polys:
                if inside_polygon(u, v, j):
                    mask[u,v] = 1

    # Segment 
 
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)        

    cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    misc.imsave('data/out/mask_' + str(ii) + '.png', mask2*255)
    #misc.imsave('data/out/' + str(ii) + '.png', mask2)



