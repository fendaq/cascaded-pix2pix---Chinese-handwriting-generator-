# -*- coding: utf-8 -*-
import argparse
import os
import PIL
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
def text2img(text, font, img_path):
    w, h = font.getsize(text)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0), font=font)
    img_array = np.array(img)
    centered_img_array = np.ones((128, 128, 3), dtype=np.uint8)*255
    h , w, d = np.where(img_array==0)
    height = max(h) - min(h)
    weight = max(w) - min(w)
    offset_h = (128 - height) //2
    offset_w = (128 - weight) //2
    #centered_img_array[offset_h:offset_h + height + 1, offset_w:offset_w + weight + 1, :]=\
    #    img_array[min(h):max(h) + 1, min(w):max(w) + 1, :]
    #centered_img = Image.fromarray(centered_img_array, 'RGB')
    #centered_img.save(img_path
    return offset_h, offset_w

parser = argparse.ArgumentParser(description="Convert text to image.")
parser.add_argument("--all", default=False, type=bool, help="generate all chinese characters. ")
parser.add_argument("--font_style", default="msjhbd.ttc")
parser.add_argument("--font_size", default=100, type=int)
parser.add_argument("--img_size", default=128, type=int)
parser.add_argument("--text", default="我")
parser.add_argument("--img_path", default=os.path.join("img","test.png"))

args = parser.parse_args()
generate_all = args.all
font_style = args.font_style
font_size = args.font_size
text = "一"
img_path = args.img_path
font_style_name = font_style.split(".")[0]
font = ImageFont.truetype(font_style, font_size)

if generate_all:
    if not os.path.isdir(os.path.join("img", "%s" % font_style_name)):
        os.mkdir(os.path.join("img", "%s" % font_style_name))
    min_h = 9999
    min_w = 9999
    for i in range(0x2E80, 0x2FDF):
        text = chr(i)
        img_path = os.path.join("img","%s" % font_style_name,\
                                "%s.png" % str(hex(i)))
        try:
            offset_h, offset_w =  text2img(text, font, img_path)
            if offset_w < min_w:
                min_w = offset_w
            if offset_h < min_h:
                min_h = offset_w
        except Exception:
            continue
    print(offset_w)
    print(offset_h)
else:
	text2img(text, font, img_path)

