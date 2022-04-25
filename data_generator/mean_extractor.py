import glob
import numpy as np
from PIL import Image

width = 128#64#32
height = 128#64#32
num_images = 0

background = Image.new('RGB', (width, height), color = "black")
bg_pixels = background.load()
#images = glob.glob("../data/32p/up/*.jpg")
#images = glob.glob("../data/64p/up/*.jpg")
#images = glob.glob("../data/64p/*.jpg")
images = glob.glob("../data/formean/*.jpg")
#images = glob.glob("../data/set6 (phasenkontrast)/64p/*.jpg")
#images = glob.glob("../data/set6 (phasenkontrast)/128p/*.jpg")
sum = np.zeros((width,height,3))
for image in images:
	with open(image, 'rb') as file:
		img = Image.open(file)
		num_images = num_images + 1
		print("image " + str(num_images))
		
		data = np.asarray(img, dtype="int32")
		#print(data.shape)
		if(data.shape == (width,height)):#greyscale image. convert to rgb
			sum[:,:,0] = sum[:,:,0] + data
			sum[:,:,1] = sum[:,:,1] + data
			sum[:,:,2] = sum[:,:,2] + data
		else:
			sum += data
		
for x in range(width):
	for y in range(height):
		r = int(sum[x,y,0]/num_images)
		g = int(sum[x,y,1]/num_images)
		b = int(sum[x,y,2]/num_images)
		#print(str(r) + " " + str(g) + " " + str(b))
		bg_pixels[y,x] = (r, g, b)

#filename = "mean" + str(width) + "up.png"
filename = "mean" + str(width) + ".png"
#filename = "mean" + str(width) + "_pc.png"
background.save(filename)
