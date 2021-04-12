#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

# Python imports
import numpy as np
import scipy.io as sio
import os, sys, time

# for clustering 
from sklearn.cluster import DBSCAN
from random import randint
import struct

# for plane removal 
from utils.lib_cloud_proc import find_plannar_points_by_grid, find_plannar_points_by_kdtree
from utils.lib_plane import PlaneModel, fit_3D_line
from utils.lib_ransac import ransac
from utils.lib_clustering import Clusterer

# ROS imports
import rospy
import ros_numpy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8

from std_msgs.msg import Header
from pyquaternion import Quaternion
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

import pdb

def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        #pdb.set_trace()
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & np.isfinite(cloud_array['intensity'])
        cloud_array = cloud_array[mask]

    # now let us transform the points 
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points 

def xyzi_array_to_pointcloud2(points_sum, msg_in):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    msg.header.stamp = msg_in.header.stamp
    msg.header.frame_id = msg_in.header.frame_id # "rs16"
    msg.height = 1 
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tobytes()
    return msg

def xyzrgb_array_to_pointcloud2(xyzrgb, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq:
            msg.header.seq = seq

        N = xyzrgb.shape[0]
        msg.height = 1
        msg.width = N
        


        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * N
        msg.is_dense = False
        msg.data = xyzrgb.tobytes()
        return msg

def band_pass_points(points, x_range= (-5, 5), y_range= (-2.2, 5.8), z_range= (-2, -0.3), i_range= (2, 8)):
    x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    i = i*255

    mean = i.mean()
    std = i.std()

    i_range = (mean + i_range[0] * std, mean + i_range[1]* std)

    # if using intensity 
    # in_range = np.logical_and.reduce((x > x_range[0], x < x_range[1],
    #        y > y_range[0], y < y_range[1],
    #        z > z_range[0], z < z_range[1],
    #        i > i_range[0], i < i_range[1]))

    # no using intensity 
    in_range = np.logical_and.reduce((x > x_range[0], x < x_range[1],
            y > y_range[0], y < y_range[1],
            z > z_range[0], z < z_range[1]))
    print("points before filtering: {0}".format(points.shape))
    points = points[in_range]
    print("points after filtering: {0}".format(points.shape))
    return points

# displaying clustering results 
def random_color_gen():
    """ Generates a random color
    
        Args: None
        
        Returns: 
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]


def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb

def split_into_columns(X): 
    ''' Split X into columns to form a list'''
    N_cols = X.shape[1]
    return [i.ravel() for i in np.hsplit(X, N_cols)]

def get_3d_line_equation(vec, p):
    ''' Get the string of 3d line equation from line parameters '''
    # vec: line direction
    # p: a point on the line
    vars = ['x', 'y', 'z']
    eqs = []
    for i in range(3):
        sign = '-' if p[i] > 0 else '+'
        s = "({}{}{:.2f})/({:.5f})".format(vars[i], sign, abs(p[i]), vec[i])
        eqs.append(s)
    return " == ".join(eqs)

# Converter 
class RS16_LaneDet_Manager():
    def __init__(self):
        self.rs16_topic = rospy.get_param('~rs16_topic', '/ns1/rslidar_points')
        self.rs32_topic = rospy.get_param('~rs32_topic', '/ns2/rslidar_points')
        # Define subscribers
        self.pc_rs16_sub = rospy.Subscriber(self.rs16_topic, PointCloud2, self.rs16_callback, queue_size = 1, buff_size = 2**24)

        # Define publishers
        rospy.loginfo("Launched node for Lidar based Lane Det.")
        # Load publisher topics
        self.filtered_pc_topic = rospy.get_param('~filtered_pc_topic', '/ns3/filtered_points')
        self.pub_rs16_filtered = rospy.Publisher(self.filtered_pc_topic, PointCloud2, queue_size=10)
        self.clustered_pc_topic = rospy.get_param('~clustered_pc_topic', '/ns3/clustered_points')
        self.pub_rs16_clustered = rospy.Publisher(self.clustered_pc_topic, PointCloud2, queue_size=10)
        # Spin
        rospy.spin()

    def rs16_callback(self, msg):
        # related data info 
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg) # msg is of type pointcloud2 
        points_np_array = get_xyzi_points(msg_cloud, True) # size: N x 4 (last row is (0, 0, 0, 1)), so it will maintrain its intensity, format: (x, y, z, intensity)
        print("points shape: {0}".format(points_np_array.shape))

        # Perform Lane Detection 
        # STEP 1 - Preprocessing: remove points that are "useless" to find road boundaries.
        # -- Road boundaries are white reflective lines. Therefore, points that are describing a road boundary 
        #         will have a high intensity value in the point cloud: this is the first filter to remove points. If a point 
        #         has an intensity value less than a threshold T=180, then this point is removed from the point cloud. 
        # -- Road boundaries are also on the ground. Therefore, points that are 5 ft above or below the ground are not interesting. 
        #         They probably describe buildings or other elements of the environment but are useless to detect the road boundaries. 
        #         This is the second filter to remove points: we consider a subset of N points and only keep points that have an elevation 
        #         between the 35th percentile and the 65th percentile for these N points. This allows us to have a dynamic threshold on elevation 
        #         by processing the whole data by small chunks.
        points_bounded = band_pass_points(points_np_array)
        msg_filtered_16 = xyzi_array_to_pointcloud2(points_bounded, msg)
        self.pub_rs16_filtered.publish(msg_filtered_16)

        # STEP 2 - Noise removal using clustering
        # In this step, the goal is to delete the noise that remains on the previous results to improve the process of line fitting.
        # Using the scikit-learn toolkit, clustering is used to remove noise. The algorithm is the following:
        # -- Divide the points in N subsets
        # -- For each subset, keep the points as if they were in the plane latitude/longitude (i.e. remove the elevation) and run a clustering algorithm
        # -- Keep the largest clusters
        # The scikit-learn toolkit several possibilities for clustering algorithms: choice amongst K-Means, Affinity Propagation, DBSCAN, Ward, and Mean Shift.
        # A right algorithm for our application must match with the following properties:
        # -- it does not take the required number of clusters as an input.
        # -- it scales well with a large amount of data.
        # These two criterions led to the use of DBSCAN as the clustering algorithm.
        db = DBSCAN(eps=1.2, min_samples=20).fit(points_bounded) # note that -1 is for outlier (noise)
        labels = db.labels_
        unique_labels = set(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Final list of labels:       ", labels)
        print("Number of Clusters:         ", n_clusters_)
        print("Number of Noise points:     ", n_noise_)
        # displaying clustering results 
        # core sample mask 
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 
        core_samples_mask[db.core_sample_indices_] = True 
        # get clour list 
        xyz_rgb_list = []
        for _, k in enumerate(unique_labels): 
            # clour for cluster k 
            if k == -1: # noise 
                color = [0, 0, 0] # black color for noise
            else: 
                color = random_color_gen()
            color = rgb_to_float(color)

            class_member_mask = (labels == k) 
            pc_label_k_member = points_bounded[class_member_mask]
            Points_Num = len(pc_label_k_member)
            # using only x, y, z (exclude intensity) 
            xyz_rgb_list.append(np.asarray([pc_label_k_member[:, 0], pc_label_k_member[:, 1], pc_label_k_member[:, 2], [color]*Points_Num], np.float32).T) 
        
        points_xyz_rgb = np.concatenate(xyz_rgb_list, axis=0)

        msg_cluster_16 = xyzrgb_array_to_pointcloud2(points_xyz_rgb, frame_id = msg.header.frame_id)
        self.pub_rs16_clustered.publish(msg_cluster_16)
        ############################################################################################################
        # STEP 3: remove non-planar region 
        ############################################################################################################
        # Get Planar regions (i.e., remove regions with large thickness in z-axis) 
        t0 = time.time()
        method = ['grid', 'kdtree'][1]

        if method == 'grid': # Not so good result, but fast
            inliers = find_plannar_points_by_grid(points_bounded, grid_size=0.5, max_height=0.1)
            # time: 1s
        elif method == 'kdtree': # Good result, but slow
            inliers = find_plannar_points_by_kdtree(points_bounded, num_neighbors=10, max_height=0.1)
            # time: 78s
            
        points_bounded_planar = points_bounded[inliers, :]

        print("\n\nTime cost of {} method: {:.2f} s".format(method, time.time()-t0))
        print("Removing non-planar regions:\n{} --> {}".format(
            points_bounded.shape[0], points_bounded_planar.shape[0]))

        # Fit road plane by RANSAC 
        points_xyz = points_bounded_planar[:,0:3]
        plane_model = PlaneModel(feature_dimension=3)
        w, inliers = ransac(
            points_xyz,
            plane_model, 
            n_pts_base=3,
            n_pts_extra=50,
            max_iter=100,
            dist_thre=0.3,
            print_time=True,
            debug=False,
        )
        print("Plane fitting: {} --> {}".format(points_xyz.shape[0], inliers.size ))
        points_bounded_plane = points_bounded_planar[inliers, :]
        # Use RANSAC to find lane direction
        points_xy = points_bounded_plane[:,0:2]
        line_model = PlaneModel(feature_dimension=2)
        w, inliers = ransac(
            points_xy,
            line_model, 
            n_pts_base=2,
            n_pts_extra=10,
            max_iter=1000,
            dist_thre=0.3,
            print_time=True,
            print_iter=False,
            debug=False,
        )
        print("Line fitting: {} --> {}".format(points_xy.shape[0], inliers.size ))
        points_bounded_plane_temp = points_bounded_plane[inliers, :]

        # Project points to normal of lane, and do clustering (Projection is same as squashing the points along the lane direction)
        # Project points
        # get line's normal from line parameters w 
        # w: w[0] + w[1]*x + w[2]*y = 0
        norm_direction = w[1:]
        print("Normal of lane: {}".format(norm_direction))

        # project points
        x = points_xy.dot(norm_direction)
        y = np.zeros_like(x)
        projections = np.column_stack((x, y))

        # Do clustering
        # settings
        min_points_in_a_lane = 10
        max_width_of_a_lane = 0.2 # (meters)
        # do clusttering
        cluster = Clusterer()
        cluster.fit(
            projections, 
            eps=max_width_of_a_lane, 
            min_samples=min_points_in_a_lane)
        print("Number of clusters (lanes): ", cluster.n_clusters)

        # STEP 4 
        # Fit a 3D line to each cluster
        N = cluster.n_clusters # Number of lanes
        lanes_param = []
        for i in range(N):
            # Get points of label i
            indices = (cluster.labels == i)
            points = points_bounded_plane[indices, :].copy()
            
            # Fit line
            x, y, z, alpha = split_into_columns(points)
            lane_direction, a_point_on_lane = fit_3D_line(x, y, z)
            lanes_param.append((lane_direction, a_point_on_lane))
            
            # Print result
            print("\n{}th line, {} points.".format(i+1, points.shape[0]))
            print("Equation: {}".format(
                get_3d_line_equation(lane_direction, a_point_on_lane)
            ))
        # Add fitted line to point cloud, and display
        # Generate data points on each of the fitted line
        points_of_each_lane = []
        for i, lane_param in enumerate(lanes_param):
            vec, p0 = lane_param
            line_pts = p0 + vec * np.mgrid[-80:80:12000j][:, np.newaxis]
            points_of_each_lane.append(line_pts)

        # Add these line points to the point cloud of the street
        lines_pts = np.vstack(points_of_each_lane)
        n = lines_pts.shape[0]
        lines_pts_with_alpha = np.hstack((lines_pts, np.ones((n, 1))))
        xyza_final = np.vstack((points_bounded, lines_pts_with_alpha))

if __name__=="__main__":
    # Initialize node
    rospy.init_node("RS16_LaneDet_Manager_Node")
    dm = RS16_LaneDet_Manager()

