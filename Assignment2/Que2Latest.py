#!/usr/bin/env python
# coding: utf-8

# In[11]:



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


# In[1]:



import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
model.eval()


# In[2]:


# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# In[3]:


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)


# In[ ]:





# In[19]:


import skimage.io as io
filename = io.imread("ImagesForAssignment2\kri2.png")
io.imshow(filename)


# In[18]:


filename.shape


# In[17]:


from PIL import Image
from torchvision import transforms
#input_image = Image.open(filename)

input_image = Image.open("ImagesForAssignment2\kri2.png")
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
# plt.show()


# In[9]:


input_image.shape


# In[14]:


image = io.imread("ImagesForAssignment2\kri2.png")
#io.imshow(image)
image = img_as_float(image)
image = rgb2gray(image)


# In[15]:


image.shape


# In[ ]:




