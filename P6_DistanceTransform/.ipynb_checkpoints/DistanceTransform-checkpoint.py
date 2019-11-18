import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from time import time
import sys

def removeChannels(image):
	height = image.shape[0]
	width = image.shape[1]
	binary_image = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			binary_image[i,j] = image[i,j,0]

	return binary_image

def thresholding(image, T):   
    arr = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    channels = 1
    
    for i in range(height):
        for j in range(width):          
            if( arr[i,j] < T):
                arr[i,j] = 0
            else:
                arr[i,j] = 255
       
    return arr

def dilateThresholding(image, T):   
    arr = image.copy()
    height = image.shape[0]
    width = image.shape[1]
    channels = 1
    
    for i in range(height):
        for j in range(width):          
            if( arr[i,j] <= T):
                arr[i,j] = 255
            else:
                arr[i,j] = 0
       
    return arr

def putSaltNoise(bimage, percent):
	height = bimage.shape[0]
	width = bimage.shape[1]

	p = int((height*width*percent)/100)

	rtn = bimage.copy()
	for i in range(p):
		x = random.randint(0, height-1)
		y = random.randint(0, width-1)
		rtn[x, y] = int(255)
	return rtn

def putPepperNoise(bimage, percent):
	height = bimage.shape[0]
	width = bimage.shape[1]

	p = int((height*width*percent)/100)

	rtn = bimage.copy()
	for i in range(p):
		x = random.randint(0, height-1)
		y = random.randint(0, width-1)
		rtn[x, y] = int(0)
	return rtn

def complement(bimage):
	return bimage^0xff

def distanceTransform_4N(bimage):
	height = bimage.shape[0]
	width = bimage.shape[1]
	offset_h = 1
	offset_w = 1

	distanceTransform = np.zeros((height+2, width+2), dtype = np.uint8)
	distanceTransform_final = np.zeros((height, width), dtype = np.uint8)

	distanceTransform[0, :] = 254
	distanceTransform[height+offset_h, :] = 254
	distanceTransform[:, 0] = 254
	distanceTransform[:, width+offset_w] = 254

	for i in range(height):
		for j in range(width):
			if(bimage[i, j] != 0):
				new_i = i+offset_h
				new_j = j+offset_w
				value = min(distanceTransform[i, new_j], distanceTransform[new_i, j])+1
				if(value > 255):
					value = 255
				distanceTransform[new_i, new_j] = value

	# print(distanceTransform)

	for i in range(height-1, -1, -1):
		for j in range(width-1, -1, -1):
			if(bimage[i,j] != 0):
				new_i = i+offset_h
				new_j = j+offset_w
				value = min(distanceTransform[new_i+1, new_j], distanceTransform[new_i, new_j+1]) + 1
				value = min(value,  distanceTransform[new_i, new_j])
				if(value > 255):
					value = 255
				distanceTransform[new_i, new_j] = value
				distanceTransform_final[i,j] = value

	# print(distanceTransform)
	return distanceTransform_final



def distanceTransform_8N(bimage):
	height = bimage.shape[0]
	width = bimage.shape[1]
	offset_h = 1
	offset_w = 1

	distanceTransform = np.zeros((height+2, width+2), dtype = np.uint8)
	distanceTransform_final = np.zeros((height, width), dtype = np.uint8)

	distanceTransform[0, :] = 255
	distanceTransform[height+offset_h, :] = 255
	distanceTransform[:, 0] = 255
	distanceTransform[:, width+offset_w] = 255

	for i in range(height):
		for j in range(width):
			if(bimage[i, j] != 0):
				new_i = i+offset_h
				new_j = j+offset_w
				value = min(distanceTransform[i, new_j], distanceTransform[new_i, j], distanceTransform[i, j], distanceTransform[i, new_j+1]) + 1
				if(value > 255):
					value = 255
				distanceTransform[new_i, new_j] = value

	# print(distanceTransform)

	for i in range(height-1, -1, -1):
		for j in range(width-1, -1, -1):
			if(bimage[i,j] != 0):
				new_i = i+offset_h
				new_j = j+offset_w
				value = min(distanceTransform[new_i+1, new_j], distanceTransform[new_i, new_j+1], distanceTransform[new_i+1, new_j+1], distanceTransform[new_i+1, j]) + 1
				value = min(value,  distanceTransform[new_i, new_j])
				if(value > 255):
					value = 255
				distanceTransform[new_i, new_j] = value
				distanceTransform_final[i,j] = value

	# print(distanceTransform)
	return distanceTransform_final

def show(image):
	plt.imshow(image, cmap='gray')
	plt.show()

if __name__ == '__main__':
	#-------------READING AND PREPARING IMAGES-------------#
	binary_image1_name = "./binary/a_binary.png"
	# binary_image2_name = "./binary/b_binary.png"
	
	binary_image1 = Image.open(binary_image1_name)
	# binary_image2 = Image.open(binary_image2_name)

	binary_image1_array = np.asarray(binary_image1)
	# binary_image2_array = np.asarray(binary_image2)

	
	bimage1 = removeChannels(binary_image1_array)
	# bimage2 = removeChannels(binary_image2_array)

	bimage1 = thresholding(bimage1, 125)
	# bimage2 = thresholding(bimage2, 125)

	bimage1_noisy = putSaltNoise(bimage1, 10)
	bimage1_noisy_pepper = putPepperNoise(bimage1, 10)
	#-------------END OF READING AND PREPARING IMAGES-------------#


	# test_image = np.array([
	# 	[1,1,1,1,1],
	# 	[1,1,1,1,1],
	# 	[1,1,0,1,1],
	# 	[1,1,1,1,1],
	# 	[1,1,1,1,1]
	# ])

	# distanceTransform = distanceTransform_4N(test_image)
	# print("\n\n4 NEIGBOURS: \n\n",distanceTransform)

	# distanceTransform = distanceTransform_8N(test_image)
	# print("\n\n8 NEIGBOURS: \n\n", distanceTransform)


	print("\nDISTANCE TRANFORM FOR 4N...\n")
	
	print("Printing original bimage1...")
	show(bimage1)


	# OPENING DOESN'T WORK
	# print("Calculating OPENING with DT to pepper image...")
	# show(bimage1_noisy)

	# # ERODE
	# print("\tCalculating DT 4n to erode...")
	# o_dt = distanceTransform_4N(bimage1_noisy)
	# print("\tCalculating erode...")
	# o_erode = thresholding(o_dt, 1)

	# # DILATE
	# print("\tCalculating complement to image...")
	# o_comp = complement(o_erode)
	# print("\tCalculating DT 4n to complement...")
	# o_dilate = distanceTransform_4N(o_comp)
	# print("\tCalculating dilate...")
	# opening = dilateThresholding(o_dilate, 1)
	
	# show(opening)


	print("Calculating CLOSING with DT to pepper image...")
	show(bimage1_noisy_pepper)

	# DILATE
	print("\tCalculating complement to image...")
	c_comp = complement(bimage1_noisy_pepper)
	print("\tCalculating DT 4n to complement...")
	c_dt = distanceTransform_4N(c_comp)
	print("\tCalculating dilate...")
	c_dilate = dilateThresholding(c_dt, 1)

	# ERODE
	print("\tCalculating DT 4n to dilate...")
	c_erode = distanceTransform_4N(c_dilate)
	print("\tCalculating erode...")
	closing = thresholding(c_erode, 1)
	
	show(closing)