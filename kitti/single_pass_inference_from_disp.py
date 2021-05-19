''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import time
from pynput import keyboard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
sys.path.append(BASE_DIR)
sys.path.append(MODEL_DIR)
sys.path.append(TRAIN_DIR)
from provider import from_prediction_to_label_format
import kitti_util as utils
import pickle
import argparse
import pypcd

MODELS = ["kitti-v1", "kitti-lite2", "nusc-lite2"]
MODEL_SEL = 0
if MODELS[MODEL_SEL] == "kitti-v1":
    modelname = "frustum_pointnets_v1"
    MODEL_PATH = os.path.join(TRAIN_DIR, 'log_v1', 'model.ckpt')
    NUM_POINT = 1024
elif MODELS[MODEL_SEL] == "kitti-lite2":
    modelname = "frustum_pointnets_lite2"
    MODEL_PATH = os.path.join(TRAIN_DIR, 'log_kitti_lite2', 'model.ckpt')
    NUM_POINT = 64
elif MODELS[MODEL_SEL] == "nusc-lite2":
    modelname = "frustum_pointnets_lite2"
    MODEL_PATH = os.path.join(TRAIN_DIR, 'log_nusc_lite2', 'model.ckpt')
    NUM_POINT = 64
MODEL = importlib.import_module(modelname)
BATCH_SIZE = 32
NUM_CHANNEL = 4
NUM_CLASSES = 2
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

img_bev_w, img_bev_h = 700, 1400
img_bev_resolution = 0.05
img_bev_w, img_bev_h = 600, 800
img_bev_resolution = 0.08
img_bev_w, img_bev_h = 600, 800
img_bev_resolution = 0.1
SHOW_BOX3D = True
SHOW_BOX_BEV = True
VIZ_MODE = 2 # 0: minimal, 1: lite, 2: full

# scan is Nx4 array
def write2pcd(scan, pcd_file):
    n_pts = scan.shape[0]
    with open(pcd_file, "w") as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z x_origin y_origin z_origin\n')
        f.write('SIZE 4 4 4 4 4 4\n')
        f.write('TYPE F F F F F F\n')
        f.write('COUNT 1 1 1 1 1 1\n')
        f.write('WIDTH ' + str(n_pts) + "\n")
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write('POINTS ' + str(n_pts) + "\n")
        f.write('DATA ascii\n')
        for i in range(n_pts):
            f.write("{} {} {} 0 0 0\n".format(scan[i, 0], scan[i, 1], scan[i, 2]))


''' det file format: type_str conf xmin ymin xmax ymax '''
def read_det_file(det_filename, min_conf=0.2):
    ''' Parse lines in 2D detection output files '''
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        d = line.rstrip().split(" ")
        category, prob = d[0], float(d[1])
        if prob<min_conf:
            continue
        # category=='Pedestrian', 'Car', 'Cyclist', ...
        type_list.append(category)
        prob_list.append(prob)
        box2d_list.append(np.array([float(coord) for coord in d[2:]]))
    return type_list, box2d_list, prob_list


def get_kitti_image(img_filename):
    return utils.load_image(img_filename)


def load_nuscenes_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 5))
    return scan[:, :4]


def get_kitti_lidar(lidar_filename, mode = 'kitti'):
    if lidar_filename.endswith(".pcd"):
        return read_from_pcd(lidar_filename)
    elif lidar_filename.endswith(".bin"):
        if mode == 'kitti':
            return utils.load_velo_scan(lidar_filename)
        else:
            return load_nuscenes_scan(lidar_filename)
    else:
        assert 0, "Unknown lidar_filename" % lidar_filename


def read_from_pcd(pcd_file):
    if os.path.exists(pcd_file):
        pc = pypcd.PointCloud.from_path(pcd_file)
        n_pts = len(pc.pc_data['x'])
        points = []
        for i in range(n_pts):
            x = pc.pc_data['x'][i]
            y = pc.pc_data['y'][i]
            z = pc.pc_data['z'][i]
            pt = [x, y, z, 0.5]  # FIXME: guess an intensity when not available
            points.append(pt)

        xyzi = np.array(points)  # Nx4
        print(xyzi.shape)
        return xyzi


def get_kitti_calibration(calib_filename):
    return utils.Calibration(calib_filename)


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more = False, clip_distance = None):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    if clip_distance:
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


class FrustumFeeder(object):
    def __init__(self, box2d_list, input_list, type_list, frustum_angle_list, prob_list, white_list):
        self.npoints = NUM_POINT
        self.box2d_list = box2d_list
        self.input_list = input_list
        self.type_list = type_list
        # frustum_angle is clockwise angle from positive x-axis
        self.frustum_angle_list = frustum_angle_list
        self.prob_list = prob_list
        self.white_list = white_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        cls_type = self.type_list[index]
        assert (cls_type in self.white_list)
        one_hot_vec = np.zeros((3))
        one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud (rotate_to_center)
        point_set = self.get_center_view_point_set(index)

        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        return point_set, rot_angle, self.prob_list[index], one_hot_vec

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


def disp_to_pc_rect(calib, disp, fl_t_bl, max_height):
    disp[disp < 0] = 0
    mask = disp > 0
    depth = fl_t_bl / (disp + 1. - mask) # this is uv_depth map (same size as image)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    pc_rect = calib.project_image_to_rect(points) # nx3 points in rect camera coord
    # pad 1 in the indensity dimension => nx6 (x, y, z, intensity, u, v)
    #valid = (-pc_rect[:, 1] < max_height)
    #pc_rect = np.concatenate([pc_rect[valid], np.ones((pc_rect[valid].shape[0], 1)), points[valid, :2]], 1)    
    pc_rect = np.concatenate([pc_rect, np.ones((pc_rect.shape[0], 1)), points[:, :2]], 1)
    return pc_rect


def extract_frustum_data_rgb_detection(det_filename, image_filename,
                                       disp_filename, calib_filename,
                                       ax,
                                       type_whitelist = ['Car'],
                                       img_height_threshold = 10,
                                       lidar_point_threshold = 10,
                                       lidar_format = 'kitti'):
    det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    if not det_type_list:
        return False, None, None, None, None, None, None, None

    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    calib_list = []  # calib data

    # Foreach 2D BBox in detection file, crop corresponding frustum_pc
    for det_idx in range(len(det_type_list)):
        #print('det idx: %d/%d' % (det_idx + 1, len(det_type_list)))
        if det_idx == 0:  # first det, load all input files
            t0 = time.time()
            calib = get_kitti_calibration(calib_filename)  # 3 by 4 matrix

            # Test calibration by 3d-to-2d projection:
            # pt_3d = np.zeros((1, 3))
            # pt_3d[0] = [1, 10, 0]
            # pts_2d = calib.project_velo_to_image(pt_3d)
            # print(pts_2d)

            disp_map = np.load(disp_filename)
            print("- B = {:.3f}".format(time.time() - t0))
            # pc_rect == nx6 (x, y, z, intensity, u, v)
            pc_rect = disp_to_pc_rect(calib, disp_map, 389.34, 2.0)
            print("- C = {:.3f}".format(time.time() - t0))

            if VIZ_MODE==2:
                img_bev = draw_BEV_lidar(pc_rect[:, :3], draw_pointcloud=True)
            elif VIZ_MODE==1:
                img_bev = draw_BEV_lidar(pc_rect[:, :3], draw_pointcloud=False)
            else:
                img_bev = draw_BEV_lidar(None)
            print("- D = {:.3f}".format(time.time() - t0))

            img = get_kitti_image(image_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("- E = {:.3f}".format(time.time() - t0))
            if VIZ_MODE>=1: # Draw 2D boxes onto image
                boxes2d = np.stack(det_box2d_list, axis=0)  # Nx4
                draw_boxes2d_on_img(boxes2d, img, ax)
            
            img_height, img_width, img_channel = img.shape
            pc_image_coord = pc_rect[:, 4:]

            # Used next: calib, pc_rect, pc_image_coord, img_fov_inds
            print("- F = {:.3f}".format(time.time() - t0))

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        pc_in_box_fov = pc_rect[box_fov_inds, :4]

        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or \
                len(pc_in_box_fov) < lidar_point_threshold:
            continue

        #if write_frustum_pcd:
        #    write2pcd(pc_in_box_fov, lidar_filename.replace('.bin', '_{:d}.pcd'.format(det_idx)))
            
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
        calib_list.append(calib)
        
    if False:
        with open(output_filename, 'wb') as fp:
            pickle.dump(box2d_list, fp)
            pickle.dump(input_list, fp)
            pickle.dump(type_list, fp)
            pickle.dump(frustum_angle_list, fp)
            pickle.dump(prob_list, fp)
            pickle.dump(calib_list, fp)

    bsize = len(input_list)
    if bsize > BATCH_SIZE:
        print('num detections > {}'.format(BATCH_SIZE))
        #assert bsize <= BATCH_SIZE, 'num detections must be <= {}'.format(BATCH_SIZE)
        return False, None, None, None, None, calib, img, img_bev
    batch_data = np.zeros((bsize, NUM_POINT, NUM_CHANNEL))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    batch_one_hot_vec = np.zeros((bsize, 3))  # for car,ped,cyc
    frustum_feeder = FrustumFeeder(box2d_list, input_list, \
                                   type_list, frustum_angle_list, prob_list, white_list=type_whitelist)
    for i in range(bsize):
        ps, rotangle, prob, onehotvec = frustum_feeder[i]
        #print(ps[np.argmin(ps[:,2])])
        batch_one_hot_vec[i] = onehotvec
        batch_data[i, ...] = ps[:, 0:NUM_CHANNEL]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob

    # print("batch_rot_angle={}".format(batch_rot_angle))
    return True, batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec, calib, img, img_bev


def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(0)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                         is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                                  heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'mask': end_points['mask'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


# Converts coordinate from pc_rect to bev_img u, v
def rect2bev_img(pt_rect):
    x, z = pt_rect[0], pt_rect[2]
    u = int(round(x / img_bev_resolution) + img_bev_w / 2)
    v = int(img_bev_h - round(z / img_bev_resolution))
    #v = int(round(x / img_bev_resolution) + img_bev_h / 2)
    #u = int(round(z / img_bev_resolution))
    return [u, v]


# pc_rect = (n,4) => x, y, z, intensity
def draw_BEV_lidar(pc_rect, draw_pointcloud = False, render = False):
    img_bev = np.zeros([img_bev_h, img_bev_w, 3], dtype=np.uint8)
    if draw_pointcloud and pc_rect is not None:
        downsample = max(int(pc_rect.shape[0] / 100000), 1)
        for pt_rect in pc_rect[::downsample, :]:
            [u, v] = rect2bev_img(pt_rect)
            if 0 <= u < img_bev_w and 0 <= v < img_bev_h:
                img_bev[v, u] = [255, 255, 255]
    if render:
        # Image.fromarray(img1).show()
        plt.imshow(img_bev)
        plt.show(block=False)
        raw_input()
        time.sleep(0.01)
        plt.clf()  # will make the plot window empty

    return img_bev


# boxes2d = (n,4) => xmin, ymin, xmax, ymax
def draw_boxes2d_on_img(boxes2d, img, ax, thickness = 2):
    img1 = np.copy(img)  # for 2d bbox
    for box2d in boxes2d:
        cv2.rectangle(img1, (int(box2d[0]), int(box2d[1])),
                      (int(box2d[2]), int(box2d[3])), (0, 255, 0), thickness)
    ax.imshow(img1)


# boxes3d = (n,8,2) => 8 corners per box with (x,y)
def draw_boxes3d_on_img(boxes3d, img, ax, color = (255, 0, 0), thickness = 2, render = True):
    boxes3d = boxes3d.astype(np.int32)
    for box3d in boxes3d:  # box3d is (8,2)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            cv2.line(img, (box3d[i, 0], box3d[i, 1]), (box3d[j, 0], box3d[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (box3d[i, 0], box3d[i, 1]), (box3d[j, 0], box3d[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (box3d[i, 0], box3d[i, 1]), (box3d[j, 0], box3d[j, 1]), color, thickness, cv2.LINE_AA)
    if render:
        ax.imshow(img)


def draw_boxes3d_on_img_bev(boxes3d_bev, img_bev, ax, color = (255, 0, 0), thickness = 2, render = True):
    for box3d_bev in boxes3d_bev:  # boxes3d_bev is (8,3)
        box3d_bev_centroid = np.zeros(3)
        box3d_bev_head = np.zeros(3)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            [u0, v0] = rect2bev_img(box3d_bev[i, :])
            [u1, v1] = rect2bev_img(box3d_bev[j, :])
            box3d_bev_centroid += box3d_bev[i, :]
            if k == 0:
                box3d_bev_head += box3d_bev[i, :]
                box3d_bev_head += box3d_bev[j, :]

            # use LINE_AA for opencv3
            cv2.line(img_bev, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)

        # Draw heading line
        box3d_bev_centroid /= 4.0
        box3d_bev_head /= 2.0
        [u0, v0] = rect2bev_img(box3d_bev_centroid)
        [u1, v1] = rect2bev_img(box3d_bev_head)

        # use LINE_AA for opencv3
        cv2.line(img_bev, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)

    if render:
        ax.imshow(img_bev)


def render_img(img):
    Image.fromarray(img).show()


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] // batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],))  # 3D box score
    valid = np.zeros((pc.shape[0],), dtype=bool)  # valid bit

    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = { \
            ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
            ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, batch_masks, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'], ops['mask'],
                      ep['heading_scores'], ep['heading_residuals'],
                      ep['size_scores'], ep['size_residuals']],
                     feed_dict=feed_dict)

        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

        valid[i * batch_size:(i + 1) * batch_size, ...] = np.any(batch_masks, axis=1)

        if False:
            # Compute scores
            batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
            batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
            mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
            mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
            heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
            size_prob = np.max(softmax(batch_size_scores), 1)  # B,
            batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
            scores[i * batch_size:(i + 1) * batch_size] = batch_scores
            # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1)  # B
    size_cls = np.argmax(size_logits, 1)  # B
    heading_res = np.array([heading_residuals[i, heading_cls[i]] \
                            for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i, size_cls[i], :] \
                          for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
           size_cls, size_res, valid  # , scores


def run_inference(sess, ops, batch_data, batch_rot_angle, batch_one_hot_vec, calib, img, img_bev, ax1, ax2,
                  verbose=False):

    batch_size = BATCH_SIZE
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    cur_batch_size = batch_data.shape[0]
    batch_data_to_feed[0:cur_batch_size, ...] = batch_data
    batch_one_hot_to_feed[0:cur_batch_size, :] = batch_one_hot_vec

    # Run one batch inference
    t0 = time.time()
    batch_output, batch_center_pred, \
    batch_hclass_pred, batch_hres_pred, \
    batch_sclass_pred, batch_sres_pred, \
    batch_valid = \
        inference(sess, ops, batch_data_to_feed,
                  batch_one_hot_to_feed, batch_size=batch_size)
    print("inference time={:.3f}".format(time.time() - t0))

    # print("center=")
    # print(batch_center_pred[:cur_batch_size, :])
    # print("heading=")
    # print(batch_hclass_pred[:cur_batch_size])
    # print("size=")
    # print(batch_sclass_pred[:cur_batch_size])
    # print("batch_score=", batch_score)
    # print("batch_valid=", batch_valid)

    # Visualize network output on image
    # boxes3d = (n,8,2)
    boxes3d = np.zeros((0, 8, 2))
    boxes3d_bev = np.zeros((0, 8, 3))
    for pick in range(cur_batch_size):
        if verbose:
            if not batch_valid[pick]:
                print('det %d: not valid' % (pick + 1))
                continue
            else:
                print('det %d: valid' % (pick + 1))
        h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(batch_center_pred[pick],
                                                                  batch_hclass_pred[pick],
                                                                  batch_hres_pred[pick],
                                                                  batch_sclass_pred[pick],
                                                                  batch_sres_pred[pick],
                                                                  batch_rot_angle[pick])

        #print("center={:.1f}, {:.1f}, {:.1f}, l/w/h={:.1f}, {:.1f}, {:.1f}, ry={:.2f}".format(tx, ty, tz, l, w, h, ry))
        # compute_projected_box3d returns (8,2)
        box3d, box3d_bev = utils.compute_projected_box3d(h, w, l, tx, ty, tz, ry, calib.P)
        boxes3d = np.append(boxes3d, box3d[None, :], axis=0)
        boxes3d_bev = np.append(boxes3d_bev, box3d_bev[None, :], axis=0)

    print("Num of good detections = {}".format(boxes3d.shape[0]))

    draw_boxes3d_on_img(boxes3d, img, ax1, render=SHOW_BOX3D)
    draw_boxes3d_on_img_bev(boxes3d_bev, img_bev, ax2, render=SHOW_BOX_BEV)

        
if __name__ == '__main__':
    # 0: kitti, 1: mule VLP32, 2: nuscenes, 3: white-2x, 4: kitti pseudo-lidar,
    # 5: sunny pseudo-lidar, 6: apollo pseudo-lidar, 7: CCSAD pseudo-lidar
    run_mode = 0
    if run_mode == 0:  # kitti
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = '/home/jhuang/repo/aanet/demo'
        allfiles = os.listdir(os.path.join(INPUT_DIR, 'left'))
        SAMPLES = [os.path.splitext(f)[0] for f in allfiles]
        #SAMPLES = ['000205']
        lidar_format = 'kitti'
    elif run_mode == 1:  # mule
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = os.path.join(ROOT_DIR, 'jhuang', 'VLP32')
        CALIB_FILE = os.path.join(INPUT_DIR, "calib.txt")
        SAMPLES = ['000610', '000267', '000072', '000220', '000100', '000010', '000040']
        lidar_format = 'mule'
    elif run_mode == 2:  # nuscenes
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Truck': 2}
        INPUT_DIR = os.path.join(ROOT_DIR, 'jhuang', 'nuscenes')
        lidar_format = 'nusc'
        CALIB_FILE = os.path.join(INPUT_DIR, "calib.txt")
        SAMPLES = [6, 13, 14, 18, 23, 27, 34, 39]
    elif run_mode == 3: # white-2x
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = os.path.join(ROOT_DIR, 'jhuang', 'white-2x')
        lidar_format = 'kitti'
        CALIB_FILE = os.path.join(INPUT_DIR, "calib.txt")
        SAMPLES = ['001060', '001080', '001330', '001545', '001680', '002055']
    elif run_mode == 4:  # kitti pseudo-lidar
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = os.path.join(ROOT_DIR, 'dataset', 'KITTI', 'object', 'training') # pseudo-lidar
        allfiles = os.listdir(os.path.join(INPUT_DIR, 'detections_2'))
        SAMPLES = [os.path.splitext(f)[0] for f in allfiles]
        lidar_format = 'kitti'
    elif run_mode == 5:  # sunny pseudo-lidar
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = '/media/jhuang/14e3e381-f8fe-43ea-b8bb-2e21cfe226dd/home/jhuang/U16/sunny_dataset'
        allfiles = os.listdir(os.path.join(INPUT_DIR, 'pseudo-lidar_velodyne'))
        SAMPLES = [os.path.splitext(f)[0] for f in allfiles]
        random.shuffle(SAMPLES)
        lidar_format = 'kitti'
    elif run_mode == 6:  # apollo pseudo-lidar
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = '/home/jhuang/Downloads/apollo_test'
        allfiles = os.listdir(os.path.join(INPUT_DIR, 'pseudo-lidar_velodyne'))
        SAMPLES = [os.path.splitext(f)[0] for f in allfiles]
        #SAMPLES = ['180116_042603602']
        random.shuffle(SAMPLES)
        lidar_format = 'kitti'
    elif run_mode == 7:  # CCSAD pseudo-lidar
        g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        INPUT_DIR = '/home/jhuang/Downloads/CCSAD/1'
        allfiles = os.listdir(os.path.join(INPUT_DIR, 'pseudo-lidar_velodyne'))
        SAMPLES = [os.path.splitext(f)[0] for f in allfiles]
        random.shuffle(SAMPLES)
        lidar_format = 'kitti'

    type_whitelist = g_type2onehotclass.keys()  # ['Car', 'Pedestrian', 'Cyclist', ...]
    batch_size = BATCH_SIZE
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)

    fig = plt.figure(constrained_layout=False, figsize=(13, 12))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.1, hspace=0.05)  # set the spacing between axes
    ax_2d = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[1, 0])
    ax_bev = fig.add_subplot(gs[:, -1])
    # _, ax = plt.subplots(1, 3)
    plt.ion()
    plt.show()
    plt.pause(0.001)

    for SAMPLE in SAMPLES:
        if run_mode == 0:
            DET_FILE = os.path.join(INPUT_DIR, 'detections_honda', "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "left", "{:s}.png".format(SAMPLE))
            DISP_FILE = os.path.join(INPUT_DIR, "pred", "{:s}_pred.npy".format(SAMPLE))
            CALIB_FILE = os.path.join(INPUT_DIR, "calib", "{:s}.txt".format(SAMPLE))
        elif run_mode == 1:
            DET_FILE = os.path.join(INPUT_DIR, "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "{:s}.png".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "{:s}.pcd".format(SAMPLE))
        elif run_mode == 2:
            DET_FILE = os.path.join(INPUT_DIR, "{:05d}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "{:05d}.png".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "{:05d}.bin".format(SAMPLE))
        elif run_mode == 3:
            DET_FILE = os.path.join(INPUT_DIR, "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "{:s}.png".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "{:s}.bin".format(SAMPLE))
        elif run_mode == 4:
            DET_FILE = os.path.join(INPUT_DIR, "detections_2", "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "image_2", "{:s}.png".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "velodyne", "{:s}.bin".format(SAMPLE))
            CALIB_FILE = os.path.join(INPUT_DIR, "calib", "{:s}.txt".format(SAMPLE))
        elif run_mode == 5:
            DET_FILE = os.path.join(INPUT_DIR, "detections_left", "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "left-image-half-size", "{:s}.jpg".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "pseudo-lidar_velodyne", "{:s}.bin".format(SAMPLE))
            CALIB_FILE = os.path.join(INPUT_DIR, "calib.txt")
        elif run_mode == 6:
            DET_FILE = os.path.join(INPUT_DIR, "detections_half_5", "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "resized_left", "{:s}.jpg".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "pseudo-lidar_velodyne", "{:s}.bin".format(SAMPLE))
            CALIB_FILE = os.path.join(INPUT_DIR, "calib_half.txt")
        elif run_mode == 7:
            DET_FILE = os.path.join(INPUT_DIR, "detections_left", "{:s}.txt".format(SAMPLE))
            IMG_FILE = os.path.join(INPUT_DIR, "left", "{:s}.png".format(SAMPLE))
            LIDAR_FILE = os.path.join(INPUT_DIR, "pseudo-lidar_velodyne", "{:s}.bin".format(SAMPLE))
            CALIB_FILE = os.path.join(INPUT_DIR, "calib.txt")

        print("Processing {}".format(SAMPLE))
        t0 = time.time()  # start time

        t_frustum = time.time()  # start time
        has_valid_dets, batch_data, batch_rot_angle, _, batch_one_hot_vec, calib, img, img_bev = \
            extract_frustum_data_rgb_detection(DET_FILE, IMG_FILE, DISP_FILE, CALIB_FILE, ax_2d,
                                               type_whitelist=type_whitelist,
                                               lidar_format=lidar_format)
        print("extract_frustum time={:.3f}".format(time.time() - t_frustum))
        if has_valid_dets:
            run_inference(sess, ops, batch_data, batch_rot_angle, batch_one_hot_vec, calib, img, img_bev, ax_3d, ax_bev)
        else:
            continue # skip sample with no detections

        print("total time={:.3f}".format(time.time() - t0))
        plt.draw()
        margin = 0.03
        plt.subplots_adjust(left=margin, right=1 - margin, top=1 - margin, bottom=margin)
        plt.pause(0.001)
        # Press any key to continue, 'ESC' to quit.
        finish = False
        with keyboard.Events() as events:
            for event in events:
                if event.key == keyboard.Key.esc:
                    finish = True
                break
        
        if finish:
            break


