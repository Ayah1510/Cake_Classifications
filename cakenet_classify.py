#!/usr/bin/env python3

import pvml
import numpy as np
import os

import sys
try:
    import PIL.Image
except ImportError:
    print("To use this script you need the `Pillow' libray")
    print("Install it with  'pip install Pillow'  or   'pip3 install Pillow'")
    sys.exit()

classes = os.listdir("images/test")
classes.sort()
IMAGENET_CLASSES = classes

# Check the command line
if len(sys.argv) < 2:
    print("USAGE: ./pvmlnet_classify IMAGE1 IMAGE2 IMAGE3 ...")
    sys.exit()

# Load the network
try:
    net = pvml.CNN.load("cakes2.npz")
except FileNotFoundError:
    print("Network definition file 'cakenet.npz' not found!")
    print("You can download it from the following address:")
    print("https://drive.google.com/file/d/1VmpvQkBk_dLsU54muiwsGcRRT_cSiYwa/view?usp=sharing")
    print()
    sys.exit()

# Load the images
filenames = sys.argv[1:]
images = []
for impath in filenames:
    im = PIL.Image.open(impath).convert("RGB")
    im = np.array(im.resize((224, 224), PIL.Image.BILINEAR))
    images.append(im / 255.0)
images = np.stack(images, 0)

# Classify the images
labels, probs = net.inference(images)

# Print the results
ii = np.argsort(-probs, 1)
for i in range(len(filenames)):
    for k in range(5):
        p = probs[i, ii[i, k]] * 100
        synset = IMAGENET_CLASSES[ii[i, k]]
        print("{}. {} ({:.1f}%)".format(k + 1, synset, p))
    print()
