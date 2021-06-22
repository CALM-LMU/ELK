#########################
# Imports
#########################

import os
import re
import sys
import json
import napari
import numpy as np
import pandas as pd
import argparse
from glob import glob
from skimage.io import imread, imsave
from napari.viewer import Viewer

from skimage.measure import regionprops

#########################
# Hotkeys
#########################

def set_hotkeys():
    @Viewer.bind_key('F', overwrite=True)
    def nxtimg(viewer):
        """Next image."""
        next_image()
            
    @Viewer.bind_key('1', overwrite=True)
    def moveto1(viewer):
        """Move selected object to layer 1."""
        origin = viewer.active_layer
        destination = viewer.layers[1]
        data_idx = origin.selected_data

        for idx in data_idx:
            destination.add(origin.data[idx])
            origin.remove_selected()

    @Viewer.bind_key('2', overwrite=True)
    def moveto2(viewer):
        """Move selected object to layer 2."""
        origin = viewer.active_layer
        destination = viewer.layers[2]
        data_idx = origin.selected_data

        for idx in data_idx:
            destination.add(origin.data[idx])
            origin.remove_selected()


    @Viewer.bind_key('3', overwrite=True)
    def moveto3(viewer):
        """Move selected object to layer 3."""
        origin = viewer.active_layer
        destination = viewer.layers[3]
        data_idx = origin.selected_data

        for idx in data_idx:
            destination.add(origin.data[idx])
            origin.remove_selected()


    @Viewer.bind_key('4', overwrite=True)
    def moveto4(viewer):
        """Move selected object to layer 4."""
        origin = viewer.active_layer
        destination = viewer.layers[4]
        data_idx = origin.selected_data

        for idx in data_idx:
            destination.add(origin.data[idx])
            origin.remove_selected()


def get_imglist(btn):
    global imglist
    imglist = glob(os.path.join(folder.value, '*.jpg')) + \
                        glob(os.path.join(folder.value, '*.tif')) + \
                        glob(os.path.join(folder.value, '*.png'))
    
    imglist = [x for x in imglist if not 'mask' in x]
    
    imglist = sorted(imglist, key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
        

def next_image(btn=None):
    global loaded
    global counter
    global imglist
    global skip
    
    if loaded:
        if counter < len(imglist):
            save_labels()
            counter += 1
        
        viewer.layers.select_all()
        viewer.layers.remove_selected()
        viewer.reset_view()
    
    if counter < len(imglist):
        label_image(folder.value)
        loaded = True
        
# def save_labels():
#     global counter
#     global imglist
#     global namelist
         
#     stack = []
#     for n, name in enumerate(namelist):
#         if name == 'budding':
#             continue
            
#         data = viewer.layers[name].data
        
#         data = np.expand_dims(data, axis=-1)
#         stack.append(data)
        
#     stack.append(np.zeros_like(data))
#     stack.append(np.zeros_like(data))
    
#     corr_mask = np.concatenate(stack, axis=-1)
#     corr_mask = corr_mask.astype(np.uint16)
    
#     data = viewer.layers['budding'].data
    
#     for sample in data:
#         x1 = int(sample[0][1])
#         y1 = int(sample[0][0])
#         x2 = int(sample[2][1])
#         y2 = int(sample[2][0])
        
#         try:
#             rps = regionprops(corr_mask[:,:,0][min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)])
#         except:
#             print(sample)
#             print(corr_mask.shape)
#             continue
        
#         areas = []
#         for rp in rps:
#             areas.append(rp.area)
            
#         sortidx = np.argsort(areas)
        
#         corr_mask[:,:,-2][corr_mask[:,:,0] == rps[sortidx[0]].label] = np.max(corr_mask[:,:,-2]) + 1
#         corr_mask[:,:,-1][corr_mask[:,:,0] == rps[sortidx[1]].label] = np.max(corr_mask[:,:,-1]) + 1

#     imsave(imglist[counter].replace('.tif', '_mask.tif').replace('train', 'train\\corrected'), corr_mask)
#     os.rename(imglist[counter], imglist[counter].replace('train', 'train\\corrected'))

def save_labels():
    global counter
    global imglist
    global namelist
         
    mask = viewer.layers['single_cell'].data
    
    things = []
    for cls, name in enumerate(namelist):
        data = viewer.layers[name].data
        
        for sample in data:
            samplelist = list(sample)
            samplelist = [list(x) for x in samplelist]
            
            thing = {'points': samplelist, 'class': cls+1}
            things.append(thing)
            
    tmpres = {'things': things}

    imsave(imglist[counter].replace('.tif', '_mask.tif').replace('train', 'train\\corrected'), mask)
    os.rename(imglist[counter], imglist[counter].replace('train', 'train\\corrected'))
    with open(imglist[counter].replace('.tif', '_detections.json').replace('train', 'train\\corrected'), 'w') as file:
        json.dump(tmpres, file)
        
def label_image(path):  
    global viewer
    global counter
    global skip
    global namelist
    global colorlist
            
    image = imread(imglist[counter])
    mask = imread(imglist[counter].replace('.tif', '_mask.tif'))
    
#     class_n = 3
    
    viewer.add_image(image)
        
#     for n in range(class_n):
#         #if n == 0:
#         viewer.add_labels(mask[:,:,n], opacity=0.3, name=namelist[n], visible=True)
#         #else:
    
    viewer.add_labels(mask, opacity=0.3, name='single_cell', visible=True)
    viewer.add_shapes(None, shape_type='path', edge_width=5, edge_color=colorlist[0], face_color=colorlist[0], opacity=0.5, name=namelist[0], visible=True)
    viewer.add_shapes(None, shape_type='path', edge_width=5, edge_color=colorlist[1], face_color=colorlist[1], opacity=0.5, name=namelist[1], visible=True)
    viewer.add_shapes(None, shape_type='path', edge_width=5, edge_color=colorlist[1], face_color=colorlist[2], opacity=0.5, name=namelist[2], visible=True)
    viewer.add_shapes(None, shape_type='path', edge_width=5, edge_color=colorlist[1], face_color=colorlist[3], opacity=0.5, name=namelist[3], visible=True)
    
    viewer.active_layer = viewer.layers[-1]
    
    return 

#########################
# Arguments
#########################

parser = argparse.ArgumentParser()
parser.add_argument('class_n', type=int, help='the total amount of classes')
parser.add_argument('path', type=str, help='The path to the images you want to label')
parser.add_argument('namelist', type=str, help='The path to the images you want to label')
args = parser.parse_args()

path = args.path
class_n = args.class_n
namelist = args.namelist

#########################
# GUI
#########################

counter = 0
loaded = False
imglist = []
# namelist = ['mating', 'budding', 'mating_2', 'mating_bud']
colorlist = ['#FF0011', '#0000FF', '#FFB60B', '#45B65B']

style = {'description_width': 'initial'}

imglist = []
viewer = napari.Viewer()
set_hotkeys()
napari.run()