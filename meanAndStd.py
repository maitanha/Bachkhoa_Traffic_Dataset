import numpy as np
import cv2
import os

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

img_dir = './training'
file_names = os.listdir(img_dir)

for file_name in file_names:
    file_path = os.path.join(img_dir, file_name)
    im = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
     
    for j in range(3):
        mean[j] += np.mean(im[:,:,j])
        
mean = (mean/len(file_names)) 
for file_name in file_names:
    file_path = os.path.join(img_dir, file_name)
    im = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])
std = np.sqrt(stdTemp/len(file_names))
print(mean, std)



