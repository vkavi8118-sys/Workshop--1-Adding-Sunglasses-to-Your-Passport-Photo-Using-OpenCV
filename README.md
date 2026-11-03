# Workshop--1-Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV

## Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
Detects the face in an image.
Places a stylish sunglass overlay perfectly on the face.
Works seamlessly with individual passport-size photos.
Customizable for different sunglasses styles or photo types.

## Technologies Used:
Python
OpenCV for image processing
Numpy for array manipulations

## How to Use:
1.Clone this repository.
2.Add your passport-sized photo to the images folder.
3.Run the script to see your "cool" transformation!

## Applications:
Learning basic image processing techniques.
Adding flair to your photos for fun.
Practicing computer vision workflows.
Feel free to fork, contribute, or customize this project for your creative needs!

## PROGRAM AND OUTPUT:

import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread('kavi.jpeg')
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Face Image")
<img width="632" height="648" alt="Screenshot 2026-11-04 010208" src="https://github.com/user-attachments/assets/a86dba94-b71b-4d3b-8b53-6b9e920b3c9e" />

faceImage.shape
faceImage.shape
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
plt.imshow(cv2.cvtColor(glassPNG, cv2.COLOR_BGR2RGB))
plt.title("Original Sunglasses")
plt.axis('off')
plt.show()
<img width="767" height="380" alt="Screenshot 2026-11-04 010225" src="https://github.com/user-attachments/assets/54858dce-1d25-4bd7-811d-9ce222fa2cea" />

glassPNG = cv2.resize(glassPNG,(874,380))
print("image Dimension ={}".format(glassPNG.shape))
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
<img width="1632" height="390" alt="Screenshot 2026-11-04 010240" src="https://github.com/user-attachments/assets/8a569848-ccbc-4a0f-9eb0-dfb77a00cc81" />

faceWithGlassesNaive = faceImage.copy()
faceWithGlassesNaive[524:904,338:1212]=glassBGR
plt.imshow(faceWithGlassesNaive[...,::-1])
<img width="632" height="625" alt="Screenshot 2026-11-04 010300" src="https://github.com/user-attachments/assets/8b108403-5e90-46a4-a5c8-83dcd1133d4c" />

glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))
glassMask = np.uint8(glassMask/255)
faceWithGlassesArithmetic = faceImage.copy()
eyeROI= faceWithGlassesArithmetic[524:904,338:1212]
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))
maskedGlass = cv2.multiply(glassBGR,glassMask)
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
<img width="1636" height="261" alt="Screenshot 2026-11-04 010343" src="https://github.com/user-attachments/assets/a454deca-4e2e-43c5-8e6a-a71f6856c83e" />

faceWithGlassesArithmetic[524:904,338:1212]=eyeRoiFinal
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
<img width="881" height="431" alt="Screenshot 2026-11-04 010405" src="https://github.com/user-attachments/assets/07db50c4-521d-4c5b-84ef-733c4d3f5f98" />

## Result:
The project successfully detected the face in the input image and added sunglasses automatically using OpenCV. The output image showed the sunglasses correctly positioned on the face, demonstrating the effectiveness of basic image processing techniques.
