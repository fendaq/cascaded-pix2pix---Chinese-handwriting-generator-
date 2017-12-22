import os
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter

def data_generator(dir, list, args):
	img_size = args.img_size
	path_a = args.path_a
	path_b = args.path_b
	path_output = args.path_output
	blur_size = args.blur_size
	crop_size = args.crop_size

	if not os.path.isdir('img/'+path_output):
		os.mkdir('img/'+path_output)
		os.mkdir('img/'+path_output+'/train')
		os.mkdir('img/'+path_output+'/test')
		os.mkdir('img/'+path_output+'/val')
		os.mkdir('img/'+path_output+'/train_clear')
	for i in list:
	# for i in range(0x4E00, 0x4E01):
		im_a = Image.open(path_a+'/'+i+'.png')
		#im_b = Image.open(path_b+'/'+i)
		im_b = Image.new('RGB', (img_size, img_size))
		im_b = im_b.crop((crop_size, crop_size, img_size-crop_size, img_size-crop_size))
		im_b = im_b.resize((img_size,img_size), Image.ANTIALIAS)
		
		im_a_blur = im_a.filter(ImageFilter.GaussianBlur(blur_size))
		im_b_blur = im_b.filter(ImageFilter.GaussianBlur(blur_size-2))
		
		new_im = Image.new('RGB', (img_size*3, img_size))
		new_im.paste(im_b, (0,0))
		new_im.paste(im_a_blur, (img_size,0))
		new_im.paste(im_b_blur, (img_size*2,0))
		new_im.save('img/'+path_output+'/'+dir+'/'+i+'.png')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Cancat image A and B")
	parser.add_argument("--path_a", default="mingliu")
	parser.add_argument("--path_b", default="img/LingWaiTC-Medium_128")
	parser.add_argument("--img_size", default=128, type=int)
	parser.add_argument("--path_output", default="concade")
	parser.add_argument("--blur_size", default=4, type=int)
	parser.add_argument("--crop_size", default=0, type=int)

	args = parser.parse_args()
	with open('poem.txt', 'r') as f:
		train_list = [line.rstrip() for line in f]
	#with open('test_list.txt', 'r') as f:
	#	test_list = [line.rstrip() for line in f]  
	#with open('val_list.txt', 'r') as f:
	#	val_list = [line.rstrip() for line in f]

	data_generator('train', train_list, args)
	#data_generator('test', test_list, args)
	# data_generator('val', val_list, args)
	# args.blur_size = args.blur_size-2
	# data_generator('train_clear', train_list, args)





