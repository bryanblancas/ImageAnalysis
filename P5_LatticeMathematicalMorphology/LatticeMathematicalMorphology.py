import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
# from time import time
# import sys

def show(image):
	plt.imshow(image, cmap='gray')
	plt.show()

def colorToGrayscale(image_array):
    height = image_array.shape[0] 
    width = image_array.shape[1]
    gs_img_arr = np.zeros((height, width),dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            G =  int( 0.2126 * float( image_array[i,j,0] )) # Red Channel 0
            G += int( 0.7152 * float( image_array[i,j,1] )) # Green Channel 1
            G += int( 0.0722 * float( image_array[i,j,2] )) # Blue Channel 2
            gs_img_arr[i,j] = G
    return gs_img_arr

def validateSize(image1, image2):
	if(image1.shape[0] != image2.shape[0]):
		return False;
	elif(image1.shape[1] != image2.shape[1]):
		return False;
	return True;

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

def putSaltNoise(image, percent):
	height = image.shape[0]
	width = image.shape[1]

	p = int((height*width*percent)/100)

	rtn = image.copy()
	for i in range(p):
		x = random.randint(0, height-1)
		y = random.randint(0, width-1)
		rtn[x, y] = int(255)
	return rtn

def putPepperNoise(image, percent):
	height = image.shape[0]
	width = image.shape[1]

	p = int((height*width*percent)/100)

	rtn = image.copy()
	for i in range(p):
		x = random.randint(0, height-1)
		y = random.randint(0, width-1)
		rtn[x, y] = int(0)
	return rtn


# def computeReflection(original, mapreflected, start):
# 	height = original.shape[0]
# 	width = original.shape[1]
# 	offset_h = height - start[0] - 1
# 	offset_w = width - start[1] - 1

# 	rtn = np.zeros((height, width), dtype=np.uint8)

# 	for key in mapreflected:
# 		values = key.split(",")
# 		h = int(values[0])
# 		w = int(values[1])
# 		rtn[h+offset_h, w+offset_w] = mapreflected[key]

# 	return rtn

def union(image1, image2):
	height = image1.shape[0]
	width = image1.shape[1]

	rtn = np.zeros((height, width), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			rtn[i,j] = max(image1[i,j], image2[i,j])

	return rtn

def intersection(image1, image2):
	height = image1.shape[0]
	width = image1.shape[1]

	rtn = np.zeros((height, width), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			rtn[i,j] = min(image1[i,j], image2[i,j])

	return rtn

def complement(image):
	height = image.shape[0]
	width = image.shape[1]

	rtn = np.zeros((height, width), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			rtn[i,j] = 255 - image[i,j]

	return rtn

def reflection(image, start):
	offset_h = start[0]
	offset_w = start[1]
	rtn = {}

	for index in np.ndindex(image.shape):
		string = str(-(index[0] - offset_h))+","+str(-(index[1] - offset_w))
		rtn[string] = image[index]

	return rtn

def translation(image, start, b):
	offset_h = start[0]
	offset_w = start[1]
	h = b[0]
	w = b[1]
	rtn = {}

	for index in np.ndindex(image.shape):
		string = str(h+(index[0] - offset_h))+","+str(w+(index[1] - offset_w))
		rtn[string] = image[index]

	return rtn

def prepareForDilate(image):
	height = image.shape[0]
	width = image.shape[1]

	rtn = np.zeros((height+2, width+2), dtype=np.uint8)

	for i in range(height):
		for j in range(width):
			rtn[i+1,j+1] = image[i,j]

	return rtn

def dilate(image, mask):
	height = image.shape[0]
	width = image.shape[1]
	
	rtn = np.zeros((height, width), dtype=np.uint8)

	pimage = prepareForDilate(image)
	print(mask)

	# DEBO DE RECORRER LAS COORDENADAS EN LUGAR DE LOS INDICES DE LA MÁSCARA 

	for i in range(height):
		for j in range(width):
			value = 0 
			new_i = i+1
			new_j = j+1

			# print(str(i)+","+str(j))

			for key in mask:
				values = key.split(",")
				h = int(values[0])
				w = int(values[1])

				prev = value

				# print("----------")
				# print("value = pimage["+str(i+h)+","+str(j+w)+"] + mask["+key+"]")
				
				value = pimage[new_i+h, new_j+w] + mask[key]
				value = max(prev, value)

			rtn[i,j] = value

	return rtn

if __name__ == '__main__':
	#-------------READING AND PREPARING IMAGES-------------#
	# grayscale_image1_name = "./grayscale/a_g.jpg"
	# grayscale_image2_name = "./grayscale/b_g.jpg"
	
	# grayscale_image1 = Image.open(grayscale_image1_name)
	# grayscale_image2 = Image.open(grayscale_image2_name)

	# grayscale_image1_array = np.asarray(grayscale_image1)
	# grayscale_image2_array = np.asarray(grayscale_image2)

	# if(validateSize(grayscale_image1_array, grayscale_image2_array) != True):
	# 	sys.exit("Images are different size "+str(grayscale_image1_array.shape)+" vs "+str(grayscale_image2_array.shape))

	# image1 = colorToGrayscale(grayscale_image1_array)
	# image2 = colorToGrayscale(grayscale_image2_array)

	# image1 = putPepperNoise(image1, 10)
	# image2 = putSaltNoise(image2, 10)
	#-------------END OF READING AND PREPARING IMAGES-------------#


	# gunion = union(image1, image2)
	# show(gunion)

	# gintersection = intersection(image1, image2)
	# show(gintersection)

	# gcomplement = complement(image1)
	# show(gcomplement)

	a = np.array([
			[5,5,5]
		], dtype=np.uint8)
	start = [0,1]

	mapreflected = reflection(a, start)
	# print(mapreflected)

	# greflected = computeReflection(a, mapreflected, start)
	# print(greflected)

	# b = [2,1]
	# gtraslated = translation(a, start, b)
	# print(gtraslated)


	image = np.array([
			[ 0, 0, 0, 0, 0],
			[ 0,10,10,10, 0],
			[ 0,10, 0,10, 0],
			[ 0,10,10,10, 0],
			[ 0, 0, 0, 0, 0]
		], dtype=np.uint8)

	gdilate = dilate(image, mapreflected)
	print(gdilate)


	#------------PRINTING IMAGES------------------#
	# f, axarr = plt.subplots(5,2, figsize=(20,20), sharey=True)
	# axarr[0,0].imshow(image1, cmap = 'gray')
	# axarr[0,1].imshow(image2, cmap = 'gray')
	# axarr[1,0].imshow(gunion, cmap = 'gray')
	# # axarr[1,1].imshow(image1, cmap = 'gray')
	# # axarr[2,0].imshow(image1, cmap = 'gray')
	# # axarr[2,1].imshow(image1, cmap = 'gray')
	# # axarr[3,0].imshow(image1, cmap = 'gray')
	# # axarr[3,1].imshow(image1, cmap = 'gray')
	# # axarr[4,0].imshow(image1, cmap = 'gray')
	# # axarr[4,1].imshow(image1, cmap = 'gray')
	# plt.show()
	#------------END OF PRINTING IMAGES-----------#