import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import sys

def colorToGrayscale(image_array):
    
    # Get height and width from image_array (color-image)
    height = image_array.shape[0] 
    width = image_array.shape[1]
    
    # New np-array with only 1 channel due to grayscale, without alpha
    # The size of the new array is the same that the color-image to not alterate the original resolution
    # grayscale_image_array = gs_img_arr
    gs_img_arr = np.zeros((height, width),dtype=np.uint8)
    
    # Double for to traversal image_array
    for i in range(0, height):
        for j in range(0, width):
            
            # Implementation of the sRGB model
            G =  int( 0.2126 * float( image_array[i,j,0] )) # Red Channel 0
            G += int( 0.7152 * float( image_array[i,j,1] )) # Green Channel 1
            G += int( 0.0722 * float( image_array[i,j,2] )) # Blue Channel 2
            
            # Assignment of the new value of determinate pixel in the grayscale-image
            gs_img_arr[i,j] = G
            
    # Return the grayscale-image
    return gs_img_arr

def validateSize(image1_array, image2_array):
	if(image1_array.shape[0] != image2_array.shape[0]):
		return False;
	elif(image1_array.shape[1] != image2_array.shape[1]):
		return False;
	return True;

def add(image1_gs, image2_gs):

	# Images size are equal, so it doesn't matter where the size is taken from 
	height = image1_gs.shape[0]
	width = image1_gs.shape[1]

	add_images = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			
			add = float(image1_gs[i,j]) + float(image2_gs[i,j])
			
			if(add > 255.0):
				add = 255.0

			add_images[i,j] = int(add)

	return add_images

def sub(image1_gs, image2_gs):

	# Images size are equal, so it doesn't matter where the size is taken from 
	height = image1_gs.shape[0]
	width = image1_gs.shape[1]

	sub_images = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			
			sub = float(image1_gs[i,j]) - float(image2_gs[i,j])
			
			if(sub < 0.0):
				sub = 0.0

			sub_images[i,j] = int(sub)

	return sub_images

def mulByScalar(image_gs, scalar):

	height = image_gs.shape[0]
	width = image_gs.shape[1]

	mul_image = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			
			mul = float(image_gs[i,j]) * scalar
			
			if(mul < 0.0):
				mul = 0.0
			elif(mul > 255.0):
				mul = 255.0

			mul_image[i,j] = int(mul)

	return mul_image

if __name__ == '__main__':

	image1_name = "./a.jpg"
	image2_name = "./b.jpg"
	
	image1 = Image.open(image1_name)
	image2 = Image.open(image2_name)

	image1_array = np.asarray(image1)
	image2_array = np.asarray(image2)
	
	if(validateSize(image1_array, image2_array) != True):
		sys.exit("Images are different size "+str(image1_array.shape)+" vs "+str(image2_array.shape))

	image1_gs = colorToGrayscale(image1_array);
	image2_gs = colorToGrayscale(image2_array);

	print(str(image1_gs.shape))
	print(str(image2_gs.shape))

	add_images = add(image1_gs, image2_gs)

	sub_images = sub(image2_gs, image1_gs)

	mul_image = mulByScalar(image1_gs, 2);

	f, axarr = plt.subplots(2,3)
	axarr[0,0].imshow(image1_gs, cmap = 'gray')
	axarr[0,1].imshow(image2_gs, cmap = 'gray')
	axarr[1,2].imshow(mul_image, cmap = 'gray')
	axarr[1,0].imshow(add_images, cmap = 'gray')
	axarr[1,1].imshow(sub_images, cmap = 'gray')
	plt.show()