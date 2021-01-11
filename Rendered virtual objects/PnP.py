import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_PnP.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####


    R = np.eye(3)
    t = np.zeros([3])
    H = est_homography(Pw[:,0:2],Pc) # calculate H
    K_H = np.dot(np.linalg.inv(K),H)
    h1 = K_H[:,0]
    h2 = K_H[:,1]
    h3 = np.cross(h1, h2)
    R_close = np.c_[h1,h2,h3]
    U,_,V = np.linalg.svd(R_close)
    eye_3 = np.eye(3)
    eye_3[2,2] = np.linalg.det(np.dot(U,np.linalg.inv(V)))
    R = U@eye_3@V  #calculate R and t
    t = K_H[:,2]/np.linalg.norm(h1)
    R = np.linalg.inv(R)
    t = -R@t
    # diag = 

    # print(H)

    ##### STUDENT CODE END #####

    return R, t
