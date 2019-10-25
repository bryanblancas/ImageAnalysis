import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import sys

def validateSize(image1, image2):
	if(image1.shape[0] != image2.shape[0]):
		return False;
	elif(image1.shape[1] != image2.shape[1]):
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

def getSet(bimage, start):
	height = bimage.shape[0]
	width = bimage.shape[1]
	offset_h = start[0]
	offset_w = start[1]
	bset = []

	for i in range(height):
		for j in range(width):
			if(bimage[i,j] == 1):
				bset.append([i - offset_h, j - offset_w])
			elif(bimage[i,j] == 255):
				bset.append([i - offset_h, j - offset_w])
	return bset

def compute_set_to_image(bimage, bset, start):
	height = bimage.shape[0]
	width = bimage.shape[1]
	offset_h = start[0]
	offset_w = start[1]
	rtn_image = np.zeros((height, width), dtype=np.uint8)

	for tupla in bset:
		i = tupla[0] + offset_h
		j = tupla[1] + offset_w
		if(not ((i < 0 or i >= height) or (j < 0 or j >= width))):
			rtn_image[i, j] = 255

	# for i in range(height):
	# 	for j in range(width):
	# 		if(not ((i + b[0] >= height) or (j + b[1] >= width))):
	# 			rtn_image[i + b[0], j + b[1]] = bimage[i,j]

	return rtn_image

def compute_set_to_image_mask(bimage, bset, start):
	height = bimage.shape[0]
	width = bimage.shape[1]
	offset_h = start[0]
	offset_w = start[1]
	rtn_image = np.zeros((height, width), dtype=np.uint8)

	for tupla in bset:
		i = tupla[0] + offset_h
		j = tupla[1] + offset_w
		if(not ((i < 0 or i >= height) or (j < 0 or j >= width))):
			rtn_image[i, j] = 1

	# for i in range(height):
	# 	for j in range(width):
	# 		if(not ((i + b[0] >= height) or (j + b[1] >= width))):
	# 			rtn_image[i + b[0], j + b[1]] = bimage[i,j]

	return rtn_image

def union(bimage1, bimage2):
	return bimage1 | bimage2

def intersection(bimage1, bimage2):
	return bimage1 & bimage2

def complement(bimage):
	return bimage^0xff

def traslation(bset, b):
	rtn_bset = []
	for i in range(len(bset)):
		tupla = bset[i]
		rtn_bset.append([tupla[0]+b[0], tupla[1]+b[1]])
	return rtn_bset

def reflection(bset):
	bset_reflected = []
	for tupla in bset:
		bset_reflected.append([tupla[0]*-1, tupla[1]*-1])
	return bset_reflected

def intersection_for_dilate(bimage, b_reflected_traslated):
	height = bimage.shape[0]
	width = bimage.shape[1]

	for tupla in b_reflected_traslated:
		if(tupla[0] < 0 or tupla[0] >= height):
			continue
		if(tupla[1] < 0 or tupla[1] >= width):
			continue
		if(bimage[tupla[0], tupla[1]] == 255):
			return True
	return False

def dilate(bimage, B_reflected):
	height = bimage.shape[0]
	width = bimage.shape[1]
	rtn_image = np.zeros((height, width), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			b_reflected_traslated = traslation(B_reflected, [i,j])
			if(intersection_for_dilate(bimage, b_reflected_traslated) == True):
				rtn_image[i, j] = 255

	return rtn_image

def subset_for_erode(bimage, b_traslated):
	height = bimage.shape[0]
	width = bimage.shape[1]

	for tupla in b_traslated:
		if(tupla[0] < 0 or tupla[0] >= height):
			return False
		if(tupla[1] < 0 or tupla[1] >= width):
			return False
		if(bimage[tupla[0], tupla[1]] != 255):
			return False
	return True

def erode(bimage, B):
	height = bimage.shape[0]
	width = bimage.shape[1]
	rtn_image = np.zeros((height, width), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			b_traslated = traslation(B, [i,j])
			if(subset_for_erode(bimage, b_traslated) == True):
				rtn_image[i, j] = 255

	return rtn_image



if __name__ == '__main__':
	#-------------READING AND PREPARING IMAGES-------------#
	binary_image1_name = "./binary/a_binary.png"
	binary_image2_name = "./binary/b_binary.png"
	
	binary_image1 = Image.open(binary_image1_name)
	binary_image2 = Image.open(binary_image2_name)

	binary_image1_array = np.asarray(binary_image1)
	binary_image2_array = np.asarray(binary_image2)

	if(validateSize(binary_image1_array, binary_image2_array) != True):
		sys.exit("Images are different size "+str(binary_image1_array.shape)+" vs "+str(binary_image2_array.shape))

	bimage1 = removeChannels(binary_image1_array)
	bimage2 = removeChannels(binary_image2_array)

	bimage1 = thresholding(bimage1, 125)
	bimage2 = thresholding(bimage2, 125)
	#-------------END OF READING AND PREPARING IMAGES-------------#


	#UNION
	bunion = union(bimage1, bimage2)


	#INTERSECTION
	bintersection = intersection(bimage1, bimage2)


	#COMPLEMENT
	bcomplement = complement(bimage2)


	#TRASLATION  (A)_b = {x | x = a+b}


		# test_image = np.array([
		# 		[255,  0,  0,  0,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,255,255,  0,  0],
		# 		[0  ,  0,  0,  0,  0],
		# 		[0  ,  0,  0,  0,  0]
		# 	])
		# start_of_image = [0,0]
		# b = [1,1]

		# test_image_set = getSet(test_image, start_of_image)
		# btraslation_set = traslation(test_image_set, b)
		# btraslation_image = compute_set_to_image(test_image, btraslation_set, start_of_image)

		# print("\n\ntraslation: "+str(b)+"\n\n")
		# print(test_image)
		# print("\n")
		# print(btraslation_image)


	b = [50, -50]
	start_of_image = [0,0]
	bimage2_set = getSet(bimage2, start_of_image)
	btraslation_set = traslation(bimage2_set, b)
	btraslation_image = compute_set_to_image(bimage2, btraslation_set, start_of_image)
	# print(btraslation_image)




	#REFLECTION A' = {-x | x in A}


		# test_image = np.array([
		# 		[0,0,0],
		# 		[0,1,1],
		# 		[1,0,1]
		# 	])
		# start_of_image = [1,1]

		# test_image_set = getSet(test_image, start_of_image)
		# breflection_set = reflection(test_image_set)
		# breflection_image = compute_set_to_image_mask(test_image, breflection_set, start_of_image)

		# print("\nreflection: \n")
		# print(test_image)
		# print("\n")
		# print(breflection_image)


	start_of_image = [88,137]
	bimage2_set = getSet(bimage2, start_of_image)
	breflection_set = reflection(bimage2_set)
	breflection_image = compute_set_to_image(bimage2, breflection_set, start_of_image)




	#DILATE A(+)B = { x | (B')_x intersection A != 0}


		# test_image = np.array([
		# 		[0  ,  0,  0,  0,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,  0,  0,  0,  0]
		# 	])
		# start_of_image = [0,0]
		# B = np.array([
		# 	[0,0,0],
		# 	[0,1,1],
		# 	[0,1,0]
		# 	])
		# start_B = [1,1]
		# # test = np.array([
		# # 		[255,255,255,0,0,0],
		# # 		[255,255,255,0,0,0],
		# # 		[255,255,255,0,0,0],
		# # 		[0,0,0,0,0,0],
		# # 		[0,0,0,0,0,0]
		# # 	])
		# # B = np.array([
		# # 	[0,0,0],
		# # 	[0,0,0],
		# # 	[0,0,1]
		# # 	])
		# # start_B = np.array([0,0])


		# # test_image_set = getSet(test_image, start_of_image)
		# B_set = getSet(B, start_B)
		# B_reflected = reflection(B_set)
		# bdilate = dilate(test_image, B_reflected)

		# print("\n\ndilate: \n\n")
		# print(test_image)
		# print("\n")
		# print(B)
		# print("\n")
		# print(bdilate)

	start_of_image = [0,0]
	B = np.array([
		[0,1,0],
		[1,1,1],
		[0,1,0]
		])
	start_B = [1,1]

	B_set = getSet(B, start_B)
	B_reflected = reflection(B_set)
	bdilate_image = dilate(bimage2, B_reflected)




	#ERODE A(-)B = {x | (B)_x subset A}


		# test_image = np.array([
		# 		[0  ,  0,  0,  0,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,255,255,255,  0],
		# 		[0  ,  0,  0,  0,  0]
		# 	])
		# start_of_image = [0,0]

		# B = np.array([
		# 	[0,0,0],
		# 	[1,1,1],
		# 	[0,0,0]
		# 	])
		# start_B = np.array([1,1])

		# B_set = getSet(B, start_B)
		# berode = erode(test_image, B_set)

		# print("\n\nerode: \n\n")
		# print(test_image)
		# print("\n")
		# print(B)
		# print("\n")
		# print(berode)

	B = np.array([
		[0,1,0],
		[1,1,1],
		[0,1,0]
		])
	start_B = [1,1]

	B_set = getSet(B, start_B)
	berode_image = erode(bimage2, B_set)




	f, axarr = plt.subplots(5,2)
	axarr[0,0].imshow(bimage1, cmap = 'gray')
	axarr[0,1].imshow(bimage2, cmap = 'gray')
	axarr[1,0].imshow(bunion, cmap = 'gray')
	axarr[1,1].imshow(bintersection, cmap = 'gray')
	axarr[2,0].imshow(bcomplement, cmap = 'gray')
	axarr[2,1].imshow(btraslation_image, cmap = 'gray')
	axarr[3,0].imshow(breflection_image, cmap = 'gray')
	axarr[3,1].imshow(bdilate_image, cmap = 'gray')
	axarr[4,0].imshow(berode_image, cmap = 'gray')
	axarr[4,1].imshow(bimage2, cmap = 'gray')
	plt.show()