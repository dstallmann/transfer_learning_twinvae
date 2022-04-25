import argparse
from PIL import Image, ImageDraw, ImageFilter
import random
from scipy.spatial import distance
import csv
import os
import numpy as np
import sys
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_case', help='either bright-field (bf) or phase-contrast (pc)', type=str, default="pc")
parser.add_argument('--ratio', help='multiplier of generated data', type=float, default=1.0)
args = parser.parse_args()

data_type = args.data_case #bf for bright-field  # pc for phase-contrast
 
fill_color = [180,180,180] if data_type=="bf" else [50,50,50] #default: 180 for bf, 50 for pc
fill_fluctuation = 30 if data_type=="bf" else 10 #default: 30 for bf, 10 for pc
edge_color = [80,80,80] if data_type=="bf" else [160,160,160] #default: bf: 80 default, 40 high_contrast, pc: 160 for 0.1 blur, 190 for 0.5 blur, 200 for 0.75 blur, 210 for 1 blur, +20 high contrast
edge_fluctuation = 50
SIZE = 128 #and height
num_circles_min = 1
num_circles_max = 31 #26 #exclusive, set to one higher than needed
r_min = 3
r_max = 8 if data_type=="bf" else 10 #default: 8 for bf, 10 for pc
crop_x_left = 26 if data_type=="bf" else 12 #default: 26 for bf, 12 for pc
crop_x_right = 20 if data_type=="bf" else 11 #default: 20 for bf, 11 for pc
crop_y_top = 12 if data_type=="bf" else 11 #default: 12 for bf, 11 for pc
crop_y_bottom = 6 if data_type=="bf" else 10 #default: 6 for bf, 10 for pc
allowedOverlap = 0 if data_type=="bf" else 1 #defautl: 0 for bf, 1 for pc
deformed_min = -2
deformed_max = 2

train_images_count = np.round(2469*args.ratio) if data_type=="bf" else np.round(3152*args.ratio) #2469 for bf train, 514 for test  # 3182 for pc train, 794 for test
test_images_count = 1000 #no constraint to match nat test data amount, but if wanted: 514 if data_type=="bf" else 792
foldername = "../data/128p_"+data_type+"_dyn"
bluriness = 0.5 #default: 0.5 for both
proportion_mod = 5

os.makedirs(foldername, exist_ok=True)
x = 0
y = 0
r = 0
created_circles = []

def get_white_noise_image(size):
	noise_map = Image.new("RGBA", (size, size), 255)
	random_grid = map(lambda x: (int(random.random() * 256),int(random.random() * 256),int(random.random() * 256)), [0] * size * size)
	noise_map.putdata(list(random_grid))
	noise_map = noise_map.convert("L") #conversion to luminosity noise
	noise_map = noise_map.convert("RGBA")
	return noise_map

def generate_images(numImages, isTrain):
	generated_images = 0
	
	filename = ""
	if isTrain:
		filename = '/train_gen.csv'
	else:
		filename = '/test_gen.csv'
		
	with open(foldername + filename, mode='w', newline='') as mycsv:
		writer = csv.writer(mycsv, delimiter=',')
		
		props = list(range(num_circles_min, num_circles_max))
		props.reverse()
		proportions = [j+proportion_mod for j in props]
		total = sum(proportions)
		
		images_folder = ""
		if isTrain:
			images_folder = "/train_gen/"
		else:
			images_folder = "/test_gen/"
		os.makedirs(foldername+images_folder, exist_ok=True)
		for f in glob.glob(foldername+images_folder+"*"):
			os.remove(f)
		
		while(generated_images < numImages):
			c = 0
			image = Image.open("../data_generator/empty128.png")
			image = image.convert("RGBA")
			draw = ImageDraw.Draw(image)
			i = 0
			 #num_circles = random.randrange(num_circles_min, num_circles_max)
			
			sel = random.randrange(1, total)
			z = 0
			while sel > 0:
				sel -= proportions[z]
				z += 1
			num_circles = z
			
			 #num_circles = random.randrange(1,466) #1+...+40=(1+40)*20=820 or 1+...+31=(1+31)*15=465? +1 for numerical reasons
			 #c = 1
			 #while num_circles > num_circles_max-c:
			 #	if num_circles_max-c < 0:
			 #		print("error in num_circles calculation")
			 #	num_circles -= (num_circles_max-c)
			 #	c += 1
			 #num_circles = c
			
			while(i < num_circles):
				r = random.randrange(r_min, r_max)
				x = random.randrange(r + crop_x_left, SIZE - r - crop_x_right, 2)
				y = random.randrange(r + crop_y_top, SIZE - r - crop_y_bottom, 2)
				accectable_position = True
				for elem in created_circles:
					dist = distance.euclidean([x,y],[elem[0],elem[1]])
					if(dist < r+elem[2]-allowedOverlap):
						accectable_position = False
						 #print("invalid position")
						break
				if(accectable_position):
					deform_x = random.randrange(deformed_min, deformed_max)
					deform_y = random.randrange(deformed_min, deformed_max)
					if r <=3:
						deform_x /= 2
						deform_y /= 2
					side_selector = random.randrange(0,1)
					if side_selector == 0:
						left = x-r+deform_x
						right = x+r
					else:
						left = x-r
						right = x+r+deform_x
					side_selector = random.randrange(0,1)
					if side_selector == 0:
						top = y-r+deform_y
						bottom = y+r
					else:
						top = y-r
						bottom = y+r+deform_y
					b = random.randrange(0, edge_fluctuation)
					e_color = np.copy(edge_color)
					e_color[0] += b
					e_color[1] += b
					e_color[2] += b
					draw.ellipse((left, top, right, bottom), fill=tuple(e_color))
					b = random.randrange(0, fill_fluctuation)
					cell_color = np.copy(fill_color)
					cell_color[0] += b
					cell_color[1] += b
					cell_color[2] += b
					draw.ellipse((left + 1, top + 1, right - 1, bottom - 1), fill=tuple(cell_color))
					i += 1
					created_circles.append([x,y,r])
			
			file = foldername + images_folder + str(SIZE) + "_" + str(generated_images) + ".png"
					
			blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
			mean = Image.open("../data_generator/mean" + str(SIZE) + "_" + data_type + ".png")
			mean = mean.convert("RGBA")
			
			 #SHOULD NOT BE REQUIRED DUE TO NOISE OPTIONS IN dataloader.py
			 #noise = get_white_noise_image(SIZE)
			 #blend = Image.blend(image, noise, 0.1) #hellfeld
			 #blend = Image.blend(image, noise, 0.0) #pc
			
			 #blend = Image.blend(image, blurred, bluriness) #hellfeld
			blend = Image.blend(image, blurred, bluriness) #pc
			blend = blend.convert("RGBA")
			
			result = Image.alpha_composite(mean, blend)
			
			result.save(file)
			writer.writerow([file, num_circles])
			generated_images += 1
			created_circles.clear()
			
			c += 1
			
generate_images(train_images_count, True)
generate_images(test_images_count, False)