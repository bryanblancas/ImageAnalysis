import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time



# Only for positive values
def nearest_integer(i):
    if( float(i-int(i)) >= 0.5 ):
        return int(i)+1
    return int(i)


# y = (x-x1)( (y2-y1) / (x2-x1) ) + y1
# For this algorithm x2-x1 = 1 always
# y = (x-x1)(y2-y1) + y1
def line(x, x1, y1, x2, y2):
    f1 = x-x1
    f2 = int(y2)-int(y1)
    y = (f1*f2) + y1
    return y


# Calculate ratio given by:
# original_size:x -> new_size:x1
# variable_a -> 1
def calculate_ratio(x, x1):
    return x/x1



def resize(image_array, new_height, new_width):
	########################################
	height = image_array.shape[0]
	width = image_array.shape[1]
	channels = image_array.shape[2]
	########################################

	# Create a copy of the image
	resized_image = image_array.copy()


	#####################
	#	NEW HEIGHT	
	#####################

	# Resize the image
	resized_image.resize(new_height, width, channels)
	print(resized_image.shape)

	# Calculate new ratio
	a = calculate_ratio(height, new_height)

	#Traversal the complete channels, the complete width and the new height
	for current_channel in range(3):
	    
	    for current_column in range(width):
	    
	        for i in range(1, new_height):

	            # Current value of new pixel
	            x = i * a
	            
	            # Values of nearest pixel to the left of X
	            x1 = int(x)
	            y1 = image_array[x1, current_column, current_channel]
	            
	            # Values of nearest pixel to the right of X
	            x2 = x1 + 1
	            if(x2 == height):
	            	x2 -= 1
	            y2 = image_array[x2, current_column, current_channel]
	            
	            # Calculate new tonal value of new pixel i
	            new_value = nearest_integer(line(x, x1, y1, x2, y2))
	            
	            # Assigment of new tonal value in the resized image
	            resized_image[i, current_column, current_channel] = new_value


	#####################
	#	NEW WIDTH	
	#####################

	# Create a new array in order of set the new width
	# Now resized_image, i.e. the image with a new height is our initial image
	resized_image2 = resized_image.copy()
	resized_image2.resize(new_height, new_width, channels)
	print(resized_image2.shape)

	# Calculate new ratio
	a = calculate_ratio(width, new_width)

	#Traversal the complete channels, the complete width and the new height
	for current_channel in range(3):
	    for current_row in range(new_height):
	        for i in range(1, new_width):
	            
	            # Current value of new pixel
	            x = i * a
	            
	             # Values of nearest pixel to the left of X
	            x1 = int(x)
	            y1 = resized_image[current_row, x1, current_channel]
	            
	            # Values of nearest pixel to the right of X
	            x2 = x1 + 1
	            if(x2 == width):
	            	x2 -= 1
	            y2 = resized_image[current_row, x2, current_channel]
	            
	             # Calculate new tonal value of new pixel i
	            new_value = nearest_integer(line(x, x1, y1, x2, y2))
	            
	            # Assigment of new tonal value in the resized image
	            resized_image2[current_row, i, current_channel] = new_value

	return resized_image2





if __name__ == '__main__':
	
	image_name = "./ampli.jpg"
	image = Image.open(image_name)
	image_array = np.asarray(image)


	start_time = time()

	new_height = int(input("New height: "))
	new_width = int(input("New width: "))
	
	resized_image = resize(image_array, new_height, new_width)

	elapsed_time = time() - start_time
	print("Algorithm time: %0.10f seconds." % elapsed_time)

	print("Original size: " + str(image_array.shape))
	print("New size: " + str(resized_image.shape))
	

	plt.subplot(2, 1, 1)
	plt.title('RESIZING AN IMAGE \n\n'+'Original image '+ str(image_array.shape)+' -> '+'Resized image '+ str(resized_image.shape))
	plt.imshow(image_array)

	plt.subplot(2, 1, 2)
	plt.imshow(resized_image)

	plt.show()


