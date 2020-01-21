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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import kitti_util as utils
import cPickle as pickle
import argparse

g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
modelname = "frustum_pointnets_v1"
MODEL = importlib.import_module(modelname)
MODEL_PATH = "train/log_v1/model.ckpt"

BATCH_SIZE = 32
NUM_POINT = 1024
NUM_CHANNEL = 4
NUM_CLASSES = 2
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type


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


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def get_kitti_image(img_filename):
    return utils.load_image(img_filename)


def get_kitti_lidar(lidar_filename):
    return utils.load_velo_scan(lidar_filename)


def get_kitti_calibration(calib_filename):
    return utils.Calibration(calib_filename)


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
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
    def __init__(self, id_list, box2d_list, input_list, type_list, frustum_angle_list, prob_list):
        self.npoints = NUM_POINT
        self.id_list = id_list
        self.box2d_list = box2d_list
        self.input_list = input_list
        self.type_list = type_list
        # frustum_angle is clockwise angle from positive x-axis
        self.frustum_angle_list = frustum_angle_list
        self.prob_list = prob_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        cls_type = self.type_list[index]
        assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
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


def extract_frustum_data_rgb_detection(det_filename, image_filename,
                                       lidar_filename, calib_filename,
                                       type_whitelist=['Car'],
                                       write_to_pcd=False,
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    # Foreach 2D BBox in detection file, crop corresponding frustum_pc
    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % \
              (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = get_kitti_calibration(calib_filename)  # 3 by 4 matrix
            pc_velo = get_kitti_lidar(lidar_filename)
            if write_to_pcd:
                write2pcd(pc_velo, lidar_filename.replace('.bin', '.pcd'))
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = get_kitti_image(image_filename)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_image_coord, img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]

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

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

    if False:
        with open(output_filename, 'wb') as fp:
            pickle.dump(id_list, fp)
            pickle.dump(box2d_list, fp)
            pickle.dump(input_list, fp)
            pickle.dump(type_list, fp)
            pickle.dump(frustum_angle_list, fp)
            pickle.dump(prob_list, fp)

    bsize = len(input_list)
    assert bsize <= BATCH_SIZE, 'num detections must be <= {}'.format(BATCH_SIZE)
    print("Num of good detections = {}".format(bsize))
    batch_data = np.zeros((bsize, NUM_POINT, NUM_CHANNEL))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    batch_one_hot_vec = np.zeros((bsize, 3))  # for car,ped,cyc
    frustum_feeder = FrustumFeeder(id_list, box2d_list, input_list, \
                                   type_list, frustum_angle_list, prob_list)
    for i in range(bsize):
        ps, rotangle, prob, onehotvec = frustum_feeder[i]
        batch_one_hot_vec[i] = onehotvec
        batch_data[i, ...] = ps[:, 0:NUM_CHANNEL]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob

    # print("batch_rot_angle={}".format(batch_rot_angle))
    return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec


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
               'end_points': end_points,
               'loss': loss}
        return sess, ops


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] / batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],))  # 3D box score

    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = { \
            ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
            ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                      ep['heading_scores'], ep['heading_residuals'],
                      ep['size_scores'], ep['size_residuals']],
                     feed_dict=feed_dict)

        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

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
           size_cls, size_res, scores


def run_inference(batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec):
    batch_size = BATCH_SIZE
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    cur_batch_size = batch_data.shape[0]
    batch_data_to_feed[0:cur_batch_size, ...] = batch_data
    batch_one_hot_to_feed[0:cur_batch_size, :] = batch_one_hot_vec

    # Run one batch inference
    batch_output, batch_center_pred, \
    batch_hclass_pred, batch_hres_pred, \
    batch_sclass_pred, batch_sres_pred, batch_scores = \
        inference(sess, ops, batch_data_to_feed,
                  batch_one_hot_to_feed, batch_size=batch_size)

    print("center=")
    print(batch_center_pred[:cur_batch_size, :])
    print("heading=")
    print(batch_hclass_pred[:cur_batch_size])
    # print("size=")
    # print(batch_sclass_pred[:cur_batch_size])


if __name__ == '__main__':
    type_whitelist = ['Car']  # ['Car', 'Pedestrian', 'Cyclist']
    write_to_pcd = False

    INPUT_DIR = os.path.join(ROOT_DIR, 'jhuang')
    SAMPLE = '001960'  # '000851' '000006', '000003'
    batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
        extract_frustum_data_rgb_detection(os.path.join(INPUT_DIR, "detections_{:s}.txt".format(SAMPLE)),
                                           os.path.join(INPUT_DIR, "{:s}.png".format(SAMPLE)),
                                           os.path.join(INPUT_DIR, "{:s}.bin".format(SAMPLE)),
                                           os.path.join(INPUT_DIR, "{:s}.txt".format(SAMPLE)),
                                           type_whitelist=type_whitelist,
                                           write_to_pcd=write_to_pcd)

    run_inference(batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec)
