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

def save_labels():
    global counter
    global imglist
    global namelist
    global boxes
    
    if boxes:
        mask = viewer.layers['mask'].data
        stack = []
        for n, name in enumerate(namelist[1:]):
            
            data = viewer.layers[name].data

            data = np.expand_dims(data, axis=-1)
            stack.append(data)

            for sample in data:
                x1 = int(sample[0][1])
                y1 = int(sample[0][0])
                x2 = int(sample[2][1])
                y2 = int(sample[2][0])

                try:
                    rps = regionprops(mask[:,:,0][min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)])
                except:
                    print(sample)
                    print(mask.shape)
                    continue

                areas = []
                for rp in rps:
                    areas.append(rp.area)

                sortidx = np.argsort(areas)

                mask[:,:,-2][mask[:,:,0] == rps[sortidx[0]].label] = np.max(mask[:,:,-2]) + 1
                mask[:,:,-1][mask[:,:,0] == rps[sortidx[1]].label] = np.max(mask[:,:,-1]) + 1

        imsave(imglist[counter].replace('.tif', '_mask.tif'), mask)

    else:
         
        mask = viewer.layers['mask'].data

        things = []
        for cls, name in enumerate(namelist[1:]):
            data = viewer.layers[name].data

            for sample in data:
                samplelist = list(sample)
                samplelist = [list(x) for x in samplelist]

                thing = {'points': samplelist, 'class': cls+1}
                things.append(thing)

        tmpres = {'things': things}

        imsave(imglist[counter].replace('.tif', '_mask.tif'), mask)
        with open(imglist[counter].replace('.tif', '_detections.json'), 'w') as file:
            json.dump(tmpres, file)
        
def label_image():  
    global viewer
    global counter
    global skip
    global namelist
    global colorlist
    global boxes
            
    image = imread(imglist[counter])
    mask = imread(imglist[counter].replace('.tif', '_mask.tif'))
    print(imglist[counter])
    
    try:
        prelabels = []
        for n in range(len(namelist)):
            prelabels.append([])
        with open(imglist[counter].replace('.tif', '_detections.json'), 'r') as file:
            jsonfile = json.load(file)
        for n in range(len(namelist)):
            for thing in jsonfile['things']:            
                if thing['class'] == n:
                    prelabels[n].append(thing['points'])
        for n in range(len(prelabels)):
            if prelabels[n] == []:
                prelabels[n] = None

    except:
        print('no json file found')
    
    try:
        mask = imread(imglist[counter].replace('.tif', '_mask.tif'))
    except:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        print('no mask found')
        
    viewer.add_image(image)
    
    colorlist = cm.get_cmap('viridis', class_n)
    
    viewer.add_labels(mask[:,:], opacity=0.3, name='mask', visible=True)
    
    if boxes:
        for n in range(class_n):
            viewer.add_shapes(None, shape_type='rectangle', edge_width=5, opacity=0.5, name=namelist[n], visible=True)
    else:
        for n in range(class_n):
            viewer.add_shapes(prelabels[n], shape_type='path', edge_width=5, opacity=0.5, name=namelist[n], visible=True)
            
    viewer.layers.selection.active = viewer.layers[-1]
        
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='The path to the images you want to label')
    parser.add_argument('n_classes', type=int, help='the total amount of classes')
    parser.add_argument('--boxes', type=bool,default= False, help='Add an empty mask')
    parser.add_argument('--namelist', nargs='+', default='', help='Optional list of class tags')
    args = parser.parse_args()

    path = args.path
    class_n = args.n_classes
    if args.namelist == '':
        namelist = list(range(class_n))
    else:
        namelist = args.namelist
    namelist = [str(x) for x in namelist]
    boxes = args.boxes
    
    imglist = get_imglist(path)
    
    counter = 0
    loaded = False

    style = {'description_width': 'initial'}

    viewer = napari.Viewer()
    set_hotkeys()
    napari.run()
