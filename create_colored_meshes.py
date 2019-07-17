import sys
sys.path.insert(0, "mano/")
import numpy as np
from webuser.smpl_handpca_wrapper_HAND_only import load_model

ids = [1,4,6,9,13,15,17,18,24,26,27,28,29,30,32,34,35,36,37,38,39,40,41,42,43,49,50]
for ii, i in enumerate(ids):

    f1 = open('data/meshes_registered/%02d_01r.obj'%i)
    cont1 = f1.readlines() 
    f1.close() 

    f2 = open('data/meshes_unregistered/%02d_01r.obj'%i)
    cont2 = f2.readlines() 
    f2.close() 

    f = open('data/out/%d.obj'%ii,'w')
         
    v1 = []
    v2 = []
    c2 = []

    for x in cont1:
        a = x[:len(x)-1].split(" ")
        if (a[0] == 'v'):
            v1.append(np.array([float(a[1]),float(a[2]),float(a[3])]))

    for x in cont2:
        a = x[:len(x)-1].split(" ")
        if (a[0] == 'v'):
            v2.append(np.array([float(a[1]),float(a[2]),float(a[3])]))
            c2.append(np.array([float(a[4]),float(a[5]),float(a[6])])) 
    
    v1 = np.vstack(v1)     
    v2 = np.vstack(v2)
    c2 = np.vstack(c2)

    for v in v1:   
        ind=np.argmin(np.sum((np.expand_dims(v,0)-v2) * (np.expand_dims(v,0)-v2),1))
        f.write("v %f %f %f %f %f %f\n"%(v[0],v[1],v[2],c2[ind,0],c2[ind,1],c2[ind,2]))
        
    for x in cont1:
        a = x[:len(x)-1].split(" ")
        if (a[0] == 'f'):
            f.write(x)
    
    f.close()    






