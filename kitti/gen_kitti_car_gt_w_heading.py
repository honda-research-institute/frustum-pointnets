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
from numpy import linalg as LA


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
    SHOW_IMG = 0
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    my_dict = dict()
    for data_idx in tqdm.tqdm(data_idx_list):
        objects = dataset.get_label_objects(data_idx)
        calib = dataset.get_calibration(data_idx)

        boxes_list = []
        for obj_idx in range(len(objects)):
            obj3d = objects[obj_idx]
            if obj3d.type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected 
            box2d = obj3d.box2d

            # Augment data by box2d perturbation
            xmin, ymin, xmax, ymax = box2d
            box_height = ymax - ymin
            if box_height < min_box_height:
                continue

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj3d, calib.P)  # (8,2), (8,3)
            if box3d_pts_2d is None:
                continue
            front0_3d = 0.5 * box3d_pts_3d[0, :] + 0.5 * box3d_pts_3d[4, :]
            front1_3d = 0.5 * box3d_pts_3d[1, :] + 0.5 * box3d_pts_3d[5, :]
            rear1_3d = 0.5 * box3d_pts_3d[2, :] + 0.5 * box3d_pts_3d[6, :]
            rear0_3d = 0.5 * box3d_pts_3d[3, :] + 0.5 * box3d_pts_3d[7, :]
            side0_3d = 0.5 * front0_3d + 0.5 * rear0_3d
            side1_3d = 0.5 * front1_3d + 0.5 * rear1_3d
            is_side0_closer = LA.norm(side0_3d[[0,2]]) < LA.norm(side1_3d[[0,2]])
            box2d_width = np.max(box3d_pts_2d, axis=0)[0] - np.min(box3d_pts_2d, axis=0)[0]
            front0 = 0.5 * box3d_pts_2d[0, :] + 0.5 * box3d_pts_2d[4, :]
            front1 = 0.5 * box3d_pts_2d[1, :] + 0.5 * box3d_pts_2d[5, :]
            rear1 = 0.5 * box3d_pts_2d[2, :] + 0.5 * box3d_pts_2d[6, :]
            rear0 = 0.5 * box3d_pts_2d[3, :] + 0.5 * box3d_pts_2d[7, :]
            width_front = abs((front1 - front0)[0])
            if is_side0_closer:
                width_side = abs((front0 - rear0)[0])
            else:
                width_side = abs((front1 - rear1)[0])
            width_rear = abs((rear1 - rear0)[0])
            ratio_front = width_front / box2d_width
            ratio_rear = width_rear / box2d_width
            ratio_side = width_side / box2d_width
            ratio_front_rear = min(max(ratio_front, ratio_rear), 1.0)

            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
            if VERBOSE:
                print("pos2d={:d} {:d}, ratio_front={:.2f}, ratio_rear={:.2f}, ratio_side={"
                      ":.2f}".format(int(box2d_center[0]), int(box2d_center[1]), \
                                     ratio_front, ratio_rear, ratio_side))

            # collect results
            boxes_list.append([xmin, ymin, xmax, ymax, obj3d.type, ratio_front_rear])

        if boxes_list:
            # add to dict
            my_dict['%06d' % (data_idx)] = boxes_list

            # Show img
            if SHOW_IMG:
                print('---')
                img = dataset.get_image(data_idx)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.show()

    with open(save_to, 'wb') as fp:
        pickle.dump(my_dict, fp)

if __name__ == '__main__':
    type_whitelist = ['Car', 'Van', 'Truck']
    kitti_dir = os.path.join(ROOT_DIR, 'dataset/KITTI/object')
    gen_data( \
        os.path.join(BASE_DIR, 'image_sets/train.txt'),
        os.path.join(kitti_dir, 'train_car_gt_w_heading.pkl'),
        'training',
        type_whitelist=type_whitelist)

    gen_data( \
        os.path.join(BASE_DIR, 'image_sets/val.txt'),
        os.path.join(kitti_dir, 'val_car_gt_w_heading.pkl'),
        'training',
        type_whitelist=type_whitelist)
