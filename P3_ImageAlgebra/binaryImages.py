import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import sys

def validateSize(image1_array, image2_array):
	if(image1_array.shape[0] != image2_array.shape[0]):
		return False;
	elif(image1_array.shape[1] != image2_array.shape[1]):
		return False;
	return True;

def removeChannels(image):
	height = image.shape[0]
	width = image.shape[1]
	binary_image = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			binary_image[i,j] = image[i,j,0]

	return binary_image

def union(image1_binary, image2_binary):
	# Images size are equal, so it doesn't matter where the size is taken from 
	height = image1_binary.shape[0]
	width = image1_binary.shape[1]

	union_images = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			union_images[i,j] = image1_binary[i,j] | image2_binary[i,j]

	return union_images

def intersection(image1_binary, image2_binary):
	# Images size are equal, so it doesn't matter where the size is taken from 
	height = image1_binary.shape[0]
	width = image1_binary.shape[1]

	intersection_images = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			intersection_images[i,j] = image1_binary[i,j] & image2_binary[i,j]

	return intersection_images

def complement(image1_binary):
	# Images size are equal, so it doesn't matter where the size is taken from 
	height = image1_binary.shape[0]
	width = image1_binary.shape[1]

	complement_image = np.zeros((height,width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			val = image1_binary[i,j]
			if(val > 127):
				val = 0
			elif(val <= 127):
				val = 255
			complement_image[i,j] = val

	return complement_image

def convolution(image_gs, mask, start):
	
	# DATA FOR MASK
	index_start_height = start[0]
	index_start_width = start[1] 

	height_mask = mask.shape[0]
	width_mask = mask.shape[1]

	# DATA FOR IMAGE
	height_image = image_gs.shape[0]
	width_image = image_gs.shape[1]

	# COMPLETING IMAGE ARRAY TO AVOID PROBLEMS IN CONVOLUTION

	leftover_y = height_mask - 1
	leftover_x = width_mask - 1

	#print("leftover_y: "+str(leftover_y))
	#print("leftover_x: "+str(leftover_x))

	new_height = height_image + 2*leftover_y
	new_width = width_image + 2*leftover_x

	#print("new_height: "+str(new_height))
	#print("new_width: "+str(new_width))

	convolution = np.zeros((new_height,new_width), dtype = np.uint8)
	#convolution2 = np.zeros((new_height,new_width), dtype = np.int8)
	convolution2 = image_gs.copy()
	print(convolution.shape)

	for i in range(leftover_y, height_image+leftover_y):
		for j in range(leftover_x, width_image+leftover_x):
			convolution[i,j] = image_gs[i-leftover_y, j-leftover_x]


	# MAKE CONVOLUTION 

	for i in range(leftover_y, height_image+leftover_y):
		for j in range(leftover_x, width_image+leftover_x):
			conv_value = 0.0
			for i_mask in range(height_mask):
				for j_mask in range(width_mask):
					if(mask[i_mask, j_mask] != 0):
						sub_value_y = i_mask - index_start_height
						sub_value_x = j_mask - index_start_width
						conv_value += float(mask[i_mask, j_mask]) * float(convolution[i -  sub_value_y, j - sub_value_x])
			
			conv_value = abs(conv_value)
			if(conv_value > 127.0):
				conv_value = 255.0
			elif(conv_value <= 127.0):
				conv_value = 0.0

			convolution2[i-leftover_y,j-leftover_x] = int(conv_value)

	return convolution2

if __name__ == '__main__':

	binary_image1_name = "./binary/a_binary.png"
	binary_image2_name = "./binary/b_binary.png"
	
	binary_image1 = Image.open(binary_image1_name)
	binary_image2 = Image.open(binary_image2_name)

	binary_image1_array = np.asarray(binary_image1)
	binary_image2_array = np.asarray(binary_image2)
	
	if(validateSize(binary_image1_array, binary_image2_array) != True):
		sys.exit("Images are different size "+str(binary_image1_array.shape)+" vs "+str(binary_image2_array.shape))

	image1_binary = removeChannels(binary_image1_array)
	image2_binary = removeChannels(binary_image2_array)

	print(str(image1_binary.shape))
	print(str(image2_binary.shape))

	union_images = union(image1_binary, image2_binary)

	intersection_images = intersection(image1_binary, image2_binary)

	complement_image = complement(image2_binary);

	mask = np.array([
			[-1.0,-1.0,-1.0],
			[-1.0, 8.0,-1.0],
			[-1.0,-1.0,-1.0]
			])

	convolution_image =  convolution(image2_binary, mask, [1,1])

	f, axarr = plt.subplots(2,3)
	axarr[0,0].imshow(image1_binary, cmap = 'gray')
	axarr[0,1].imshow(image2_binary, cmap = 'gray')
	axarr[0,2].imshow(convolution_image, cmap = 'gray')
	axarr[1,0].imshow(union_images, cmap = 'gray')
	axarr[1,1].imshow(intersection_images, cmap = 'gray')
	axarr[1,2].imshow(complement_image, cmap = 'gray')
	plt.show()