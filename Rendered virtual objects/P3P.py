import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_P3P.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    
    # define a,b,c,alpha,beta,gamma
    Pw_x_y = Pw[:,0:2]
    a = np.linalg.norm(Pw_x_y[0,:]-Pw_x_y[2,:])
    b = np.linalg.norm(Pw_x_y[1,:]-Pw_x_y[2,:])
    c = np.linalg.norm(Pw_x_y[0,:]-Pw_x_y[1,:])
    f = (K[0,0]+K[1,1])/2 #calculate f
    u_0 = K[0,2]
    v_0 = K[1,2]
    Pc_1 = np.array([Pc[1,0]-u_0,Pc[1,1]-v_0,f])
    Pc_2 = np.array([Pc[0,0]-u_0,Pc[0,1]-v_0,f])
    Pc_3 = np.array([Pc[2,0]-u_0,Pc[2,1]-v_0,f])
    Pw_1 = Pw[1,:]
    Pw_2 = Pw[0,:]
    Pw_3 = Pw[2,:]
    j_1 = 1/np.linalg.norm(Pc_1)*Pc_1
    j_2 = 1/np.linalg.norm(Pc_2)*Pc_2
    j_3 = 1/np.linalg.norm(Pc_3)*Pc_3
    alpha = np.arccos(j_2@j_3)
    beta = np.arccos(j_1@j_3)
    gamma = np.arccos(j_1@j_2)
    
    # define coefficients of the 4th degree polynomial
    a_cb = (a**2-c**2)/b**2
    acb = (a**2+c**2)/b**2
    b_cb = (b**2-c**2)/b**2
    b_ab = (b**2-a**2)/b**2
    A_4 = (a_cb-1)**2-4*c**2/b**2*(np.cos(alpha)**2)
    A_3 = 4*(a_cb*(1-a_cb)*np.cos(beta)-(1-acb)*np.cos(alpha)*np.cos(gamma)
        +2*c**2/b**2*np.cos(alpha)**2*np.cos(beta))
    A_2 = 2*(a_cb**2-1+2*(a_cb)**2*np.cos(beta)**2+2*b_cb*np.cos(alpha)**2
        -4*acb*np.cos(alpha)*np.cos(beta)*np.cos(gamma)+2*b_ab*np.cos(gamma)**2)
    A_1 = 4*(-a_cb*(1+a_cb)*np.cos(beta)+2*a**2/b**2*np.cos(gamma)**2*np.cos(beta)
        -(1-acb)*np.cos(alpha)*np.cos(gamma))
    A_0 = (1+a_cb)**2-4*a**2/b**2*np.cos(gamma)**2
    coeff = [A_4,A_3,A_2,A_1,A_0]
    v_4 = np.roots(coeff)  # get the get the roots of the polynomial

    # calculate real roots u and v
    v_2 = v_4[np.isreal(v_4)]

    # check for valid distances
    v = v_2[0]
    u = ((-1+a_cb)*v**2-2*a_cb*np.cos(beta)*v+1+a_cb)/(2*(np.cos(gamma)-v*np.cos(alpha)))
    
    # calculate 3D coordinates in Camera frame
    s_1 = np.sqrt(c**2/(1+u**2-2*u*np.cos(gamma)))
    s_2 = u*s_1
    s_3 = v*s_1
    A21 = s_1*j_1
    A31 = s_2*j_2
    A41 = s_3*j_3
    A = np.concatenate(([A21],[A31],[A41]))
    B21 = Pw_1
    B31 = Pw_2
    B41 = Pw_3
    B = np.concatenate(([B21],[B31],[B41]))

    # Calculate R,t using Procrustes
    R,t = Procrustes(A,B)
    ##### STUDENT CODE END #####

    return R, t

def Procrustes(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    A = X
    B = Y
    A_centroids = np.sum(A,axis=0)/3
    A = A-A_centroids
    B_centroids = np.sum(B,axis=0)/3
    B = B-B_centroids
    A = A.T
    B = B.T
    U, _, VT = np.linalg.svd(B@np.transpose(A))
    V = VT.T
    UT = U.T
    eye_3 = np.eye(3)
    eye_3[2,2] = np.linalg.det(V@UT)
    R = V@eye_3@UT

    t = A_centroids-R@B_centroids
    R = R.T
    t = -R@t   
    ##### STUDENT CODE END #####
    
    return R, t
