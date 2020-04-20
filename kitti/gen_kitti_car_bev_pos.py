''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm  # for progress bar
import cPickle as pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import kitti_util as utils
from kitti_object import *


def gen_data(idx_filename, save_to, split, type_whitelist, min_box_height = 20):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None
    '''

    VERBOSE = 0
    REMOVE_REAR_OBJ = 1 # remove objects behind velo
    SAVE_PKL = 1
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    my_dict = dict()
    if VERBOSE:
        data_idx_range = data_idx_list
    else:
        data_idx_range = tqdm.tqdm(data_idx_list)

    for data_idx in data_idx_range:
        objects = dataset.get_label_objects(data_idx)
        calib = dataset.get_calibration(data_idx)

        ann_list = []
        for obj_idx in range(len(objects)):
            obj3d = objects[obj_idx]
            if obj3d.type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected 
            obj_ctr_cam = obj3d.t # location (x,y,z) in camera coord
            obj_ctr_cam = np.array(obj_ctr_cam)[np.newaxis, ...] # convert to numpy nx3
            obj_ctr_vel = calib.project_ref_to_velo(obj_ctr_cam)[0] # velo coord

            # Remove objects behind velo
            if REMOVE_REAR_OBJ and obj_ctr_vel[0] < 0:
                continue

            if VERBOSE:
                print("idx={:06d}, center={:.1f} {:.1f}".format(data_idx, obj_ctr_vel[0], obj_ctr_vel[1]))

            # collect results (x==forward, y==left, type)
            ann_list.append([obj_ctr_vel[0], obj_ctr_vel[1], obj3d.type])

        if ann_list:
            # add to dict
            my_dict['%06d' % (data_idx)] = ann_list

    if SAVE_PKL:
        with open(save_to, 'wb') as fp:
            pickle.dump(my_dict, fp)

if __name__ == '__main__':
    type_whitelist = ['Car', 'Van', 'Truck']
    kitti_dir = os.path.join(ROOT_DIR, 'dataset/KITTI/object')
    gen_data( \
        os.path.join(BASE_DIR, 'image_sets/train.txt'),
        os.path.join(kitti_dir, 'train_car_gt_bev.pkl'),
        'training',
        type_whitelist=type_whitelist)

    gen_data( \
        os.path.join(BASE_DIR, 'image_sets/val.txt'),
        os.path.join(kitti_dir, 'val_car_gt_bev.pkl'),
        'training',
        type_whitelist=type_whitelist)
