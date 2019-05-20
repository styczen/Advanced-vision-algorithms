
import numpy as np
import matplotlib.pyplot as plt

def appendimages(im1,im2):    
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    
    return np.concatenate((im1,im2), axis=1)    
    
    
def plot_matches(im1,im2,matches):
    colors=['r','g','b','c','m','y']
    im3 = appendimages(im1,im2)
    
    plt.figure()
    plt.imshow(im3)
    
    cols1 = im1.shape[1]
    for i, m in enumerate(matches):
            plt.plot([m[0][1],m[1][1]+cols1],[m[0][0],m[1][0]],colors[i%6], linewidth=0.5)
    plt.axis('off') 