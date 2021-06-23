#########################
# Imports
#########################

import os
import re
import json
import napari
import argparse
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread, imsave
from napari.viewer import Viewer
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.measure import regionprops

#########################
# Hotkeys
#########################

def set_hotkeys():
    @Viewer.bind_key('F', overwrite=True)
    def nxtimg(viewer):
        """Next image."""
        next_image()

def get_imglist(path):
    imglist = glob(os.path.join(path, '*.jpg')) + \
                    glob(os.path.join(path, '*.tif')) + \
                    glob(os.path.join(path, '*.png'))
    imglist = [x for x in imglist if not 'mask' in x]
    imglist = sorted(imglist, key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
    return imglist


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
        label_image()
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
        
def label_image():  
    global viewer
    global counter
    global skip
    global namelist
    global colorlist
            
    image = imread(imglist[counter])
    mask = imread(imglist[counter].replace('.tif', '_mask.tif'))
        
    viewer.add_image(image)
    
    colorlist = cm.get_cmap('viridis', class_n)
        
    for n in range(class_n):
        x = colorlist(n)
        y = x[:3]
        viewer.add_labels(mask[:,:], opacity=0.3, name=namelist[n], visible=True)
        viewer.add_shapes(None, shape_type='path', edge_width=5,  face_color=y, opacity=0.5, name=namelist[n], visible=True)
    viewer.active_layer = viewer.layers[-1]
    
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='The path to the images you want to label')
    parser.add_argument('n_classes', type=int, help='the total amount of classes')
    parser.add_argument('--mask', nargs='+', default=[], help='Add an empty mask')
    args = parser.parse_args()

    path = args.path
    class_n = args.n_classes
    
    emp_c = [''] * class_n
    parser.add_argument('--namelist', nargs='+', default=emp_c, help='Optional list of class tags')
    args = parser.parse_args()
    namelist = args.namelist

    imglist = get_imglist(path)
    
    counter = 0
    loaded = False

    style = {'description_width': 'initial'}

    viewer = napari.Viewer()
    set_hotkeys()
    napari.run()
