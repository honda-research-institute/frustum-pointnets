''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import tqdm  # for progress bar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import kitti_util as utils
from kitti_object import *


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def hist_percent(data, title, ax=None):
    if not ax:
        ax = plt.gca()

    ax.hist(data, 20, weights=np.ones(len(data)) / len(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)


def analyze_frustum_data(idx_filename, split, type_whitelist=['Car']):
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
    DIST_MIN, DIST_MAX = 10, 50
    DIST_PER_BIN = 10
    DIST_BINS = list(range(DIST_MIN, DIST_MAX, DIST_PER_BIN))
    N_BINS = len(DIST_BINS)
    pt_cnt_stats = []
    for x in range(N_BINS):
        pt_cnt_stats.append([])

    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    for data_idx in tqdm.tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)  # Nx4
        pc_rect = np.zeros_like(pc_velo)
        # Convert from lidar frame to camera frame
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            obj3d = objects[obj_idx]
            if obj3d.type not in type_whitelist:
                continue

            dist = np.hypot(obj3d.t[0], obj3d.t[2]) # obj3d.t == (x,y,z) in camera coord.
            bin = int(round((dist - DIST_MIN) / float(DIST_PER_BIN)))
            if bin < 0 or bin >= N_BINS:  # distance not in range
                continue

            # 2D BOX: Get pts rect backprojected 
            box2d = obj3d.box2d

            # Augment data by box2d perturbation
            xmin, ymin, xmax, ymax = box2d
            box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                           (pc_image_coord[:, 0] >= xmin) & \
                           (pc_image_coord[:, 1] < ymax) & \
                           (pc_image_coord[:, 1] >= ymin)
            box_fov_inds = box_fov_inds & img_fov_inds
            pc_in_box_fov = pc_rect[box_fov_inds, :]

            # 3D BOX: Get pts velo in 3d box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj3d, calib.P)
            _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
            label = np.zeros((pc_in_box_fov.shape[0]))
            label[inds] = 1

            # collect statistics
            n_pts_obj = np.sum(label)
            pt_cnt_stats[bin].append(n_pts_obj)

    # the histogram of the data
    _, ax = plt.subplots(2, 2)
    data = pt_cnt_stats[0]
    hist_percent(data, 'dist=15m', ax=ax[0, 0])
    data = pt_cnt_stats[1]
    hist_percent(data, 'dist=25m', ax=ax[0, 1])
    data = pt_cnt_stats[2]
    hist_percent(data, 'dist=35m', ax=ax[1, 0])
    data = pt_cnt_stats[3]
    hist_percent(data, 'dist=45m', ax=ax[1, 1])
    plt.show()


if __name__ == '__main__':
    type_whitelist = ['Car'] # ['Car', 'Pedestrian', 'Cyclist']
    analyze_frustum_data( \
        os.path.join(BASE_DIR, 'image_sets/train.txt'),
        'training',
        type_whitelist=type_whitelist)
