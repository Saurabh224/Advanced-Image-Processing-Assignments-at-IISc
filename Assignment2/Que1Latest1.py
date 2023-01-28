#!/usr/bin/env python
# coding: utf-8

# In[2]:





import numpy as np
import skimage.io as io
import numpy.linalg as linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from numpy import linalg as LA
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean






def NormalizedCuts(image):

   def calculateWeights(x,y,xp,yp):
    # Weight matrix
    
    sigmaX = 15
    sigmaI = 10    
    n = LA.norm(x-y)
    spatial = np.exp(-n**2/sigmaI)
    n = LA.norm(xp-yp)
    intense = np.exp(-n**2/sigmaX)    
    return spatial*intense
    
   
   r,c = image.shape
   rc= r*c
   pixelval = []
   for i in range(r):
        for j in range(c):
            val = [image[i, j], i, j]
            pixelval.append(val)    
   W = np.zeros((rc,rc))   
   for i in range(rc):
     x = pixelval[i][0]
     xp = np.array([pixelval[i][1], pixelval[i][2]])
     for j in range(rc):
         y = pixelval[j][0]
         yp = np.array([pixelval[i][1], pixelval[i][2]])
         w = calculateWeights(x,y,xp,yp)
         W[i, j] = w
               
   return W

def custom_sim_img(img_i: float, img_j: float, vec_i: np.array, vec_j: np.array) -> float:

    """

    Calculate similarity between two nodes, here the nodes are image pixel

    """

    #Parameters

    sigma_i, sigma_x, r = 10, 15, 30

    feature_similarity = -((img_i-img_j) ** 2)/sigma_i

    spatial_similarity = -((np.linalg.norm(vec_i-vec_j, ord=np.inf)) ** 2)/sigma_x

    # return np.exp(feature_similarity)

   

    if np.linalg.norm(vec_i-vec_j, ord=2) < r:

        return np.exp(feature_similarity) * np.exp(spatial_similarity)


def NormalizedCuts2(image):
    
   p,q=0,0
   r,c = image.shape
   rc= r*c
   #print(r,c)
   pixelval = []
   for i in range(r):
        for j in range(c):
            val = [image[i, j], i, j]
            pixelval.append(val)    
   W = np.zeros((rc,rc))   
   for i in range(rc):
     x = pixelval[i][0]
     xp = np.array([pixelval[i][1], pixelval[i][2]])
     for j in range(rc):
         y = pixelval[j][0]
         yp = np.array([pixelval[i][1], pixelval[i][2]])
         w = custom_sim_img(x,y,xp,yp)
         W[i, j] = w
               
   return W



def performSegmentation(W):
  D = np.zeros(W.shape)
  D = np.diag(W.sum(axis=1))
  eigenValues, eigenVectors = eigsh(D-W, k=2, M=D, which="SM")
  # idx = eigenValues.argsort()[::1]   
  eigenValues = eigenValues[1]
  eigenVectors = eigenVectors[:,1]
  medn_val = np.median(eigenVectors)
  out = np.reshape(eigenVectors, image.shape)
  out = np.where(out > medn_val,1,0)
  return out
  


# Resize an Image

image = io.imread("input2.jpg")
#io.imshow(image)
image = img_as_float(image)
image = rgb2gray(image)
plt.title("Original Image")
io.imshow(image)
plt.show()

image_resized = resize(image, (64,64))
io.imshow(image_resized)
r,c = image_resized.shape
image = image_resized
W = NormalizedCuts(image)
out = performSegmentation(W)
plt.title("Segmentated Image")
io.imshow(out,cmap='gray')
plt.show()


# In[76]:


image = io.imread("kri.png")
#io.imshow(image)
image = img_as_float(image)
image = rgb2gray(image)
plt.title("Original Image")
io.imshow(image)
plt.show()

image_resized = resize(image, (64,64))
io.imshow(image_resized)
r,c = image_resized.shape
image = image_resized
W = NormalizedCuts(image)
out = performSegmentation(W)
plt.title("Segmentated Image")
io.imshow(out,cmap='gray')
plt.show()





# In[3]:


W2 = NormalizedCuts2(image)
out2 = performSegmentation(W)
plt.title("Segmentated Image")
io.imshow(out2,cmap='gray')
plt.show()


# In[ ]:




