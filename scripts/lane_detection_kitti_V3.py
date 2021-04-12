#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

# Python imports
import numpy as np
import scipy.io as sio
import os, sys, time
import math
from easydict import EasyDict
from functools import reduce 

from random import randint
import struct

import pcl 

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

import time 
# optimize python code 
from numba import njit, prange

from math import sqrt 

import pdb

def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        #pdb.set_trace()
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & np.isfinite(cloud_array['i'])
        cloud_array = cloud_array[mask]

    # now let us transform the points 
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['i']

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
        PointField('i', 12, PointField.FLOAT32, 1)
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

####################################################################################
# define common functions 
####################################################################################
def write_cloud_and_figure(filename, cloud):
    write_cloud(filename + '.pcd', cloud)
    plt.savefig(filename + '.jpg')

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
    return " == ".join(eqs) # line in symmetric equations 

def band_pass_points(points, x_range= (-40, 40), y_range= (-30, 30), z_range= (-1.5, 0.6), i_range_nvar= (2, 8)):
    x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    # i = i*255
    # mean = i.mean()
    # std = i.std()

    # i_range = (mean + i_range_nvar[0] * std, mean + i_range_nvar[1]* std)

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
    print("points after filtering (not using intensity): {0}".format(points.shape))
    return points

def organize_kitti_by_scanIDs(points_np_array):
    Points_By_Scan_ID = []
    # appened scan id 
    N = points_np_array.shape[0] # number of points, format: N x 4 (x, y, z, i)
    scan_ID = 0 
    num_scan_rings = 64 
    # compute angle 
    ori_all = np.arctan2(points_np_array[:, 1], points_np_array[:, 0]) * 180 / math.pi 
    ori_lt_zero_mask = ori_all < 0 
    ori_all[ori_lt_zero_mask] += 360
    
    for k in list(range(1,N)): 
        # y/x
        ori = ori_all[k]  
        ori_pre = ori_all[k-1]  

        if abs(ori-ori_pre) > 250: 
            scan_ID += 1
        if scan_ID < num_scan_rings:
            point_temp = points_np_array[k]
            point_temp[3] = scan_ID # set its intensity to its scanid 
            Points_By_Scan_ID.append(point_temp) 
    Points_By_Scan_ID = np.vstack(Points_By_Scan_ID)
    return Points_By_Scan_ID

def organize_kitti_by_scanIDs_fast(points_np_array):
    # appened scan id 
    N = points_np_array.shape[0] # number of points, format: N x 4 (x, y, z, i)
    scan_ID = 0 
    num_scan_rings = 64 
    # compute angle 
    ori_all = np.arctan2(points_np_array[:, 1], points_np_array[:, 0]) * 180 / math.pi 
    ori_lt_zero_mask = ori_all < 0 
    ori_all[ori_lt_zero_mask] += 360
    ori_1_to_N_1 = ori_all[1:]
    ori_0_to_N_2 = ori_all[:-1] 
    delta = np.abs(ori_1_to_N_1 - ori_0_to_N_2)
    detla_mask = delta > 250 
    scanIDs = np.cumsum(detla_mask)  
    mask_ScanIDs = num_scan_rings >= scanIDs 
    # set it 
    Points_By_Scan_ID = points_np_array[1:][mask_ScanIDs] 
    Points_By_Scan_ID[:, 3] = scanIDs[mask_ScanIDs]

    return Points_By_Scan_ID

def extract_ground_non_ground(Points_By_Scan_ID):
    # 分段进行平面分割 [-40, 40]
    # 6 Segments 
    # 7 ms
    Point_Seg_Dict = {}
    # [-15, 15]
    # 0 =< x =< 15 , |y| < 30 
    Point_Seg_By_X0 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= 0, 15 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[0] = Point_Seg_By_X0 
    # -15 =< x < 0 , |y| < 30
    Point_Seg_By_X1 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -15, Points_By_Scan_ID[:,0] < 0, abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[1] = Point_Seg_By_X1
    # (15, 30] and [-30, -15)
    # 15 < x =< 30 , |y| < 30 
    Point_Seg_By_X2 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] > 15, 30 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[2] = Point_Seg_By_X2  
    # -30 =< x < -15 , |y| < 30
    Point_Seg_By_X3 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -30, Points_By_Scan_ID[:,0] < -15, abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[3] = Point_Seg_By_X3
    # (30, 40] and [-40, -30)
    # 30 < x =< 40 , |y| < 30 
    Point_Seg_By_X4 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] > 30, 40 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[4] = Point_Seg_By_X4
    # -40 =< x < -30 , |y| < 30  
    Point_Seg_By_X5 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -40, -30 > Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
    Point_Seg_Dict[5] = Point_Seg_By_X5  
    # print('takes {0} ms.'.format((time.time()-start)*1000))
    # fit plane in each segment region
    _seg_distance_thres = 0.35   
    Ground_Points = [] 
    NoGround_Points = [] 
   
    for seg_id in range(len(Point_Seg_Dict)-2): # only use seg 0, 1, 2, 3 
        Points_of_seg_id = Point_Seg_Dict[seg_id]
        Points_of_seg_id_XYZ = Points_of_seg_id[:, :3] # M x 3 

        # Fit road plane on the points of this segment by RANSAC 
        pcl_data = pcl.PointCloud(Points_of_seg_id_XYZ.astype(np.float32))
        seg = pcl_data.make_segmenter() 
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # max distance for a point to be considered fitting the model 
        seg.set_distance_threshold(_seg_distance_thres) 
        seg.set_MaxIterations(800)  # in the original paper, it is set to 800
        seg.set_optimize_coefficients(True) 

        # below takes too much time 
        # indices correspond to points on surface (ax + by + cz + d = 0)
        # coefficients: a = coefficients[0], b = coefficients[1], c = coefficients[2], d = coefficients[3]
        plane_points_indices, coefficients = seg.segment()

        plane_points_mask = np.zeros((Points_of_seg_id.shape[0]), dtype=bool)
        plane_points_mask[plane_points_indices] = True 
        plane_points = Points_of_seg_id[plane_points_mask] 
        
        # Extract points that donot fit the RANSAC plane model
        non_plane_points_mask = np.logical_not(plane_points_mask)
        non_plane_points = Points_of_seg_id[non_plane_points_mask]

        Ground_Points.append(plane_points) 
        NoGround_Points.append(non_plane_points)

    Ground_Points = np.vstack(Ground_Points)
    NoGround_Points = np.vstack(NoGround_Points)

    return Ground_Points, NoGround_Points


def extract_ground_non_ground_fast(Point_Segs_By_X):
    _seg_distance_thres = 0.35   
    Ground_Points = [] 
    NoGround_Points = [] 
    for seg_id in range(len(Point_Segs_By_X)): # only use seg 0, 1, 2, 3
        Points_of_seg_id = Point_Segs_By_X[seg_id] 
        Points_of_seg_id_XYZ = Points_of_seg_id[:, :3] # M x 3 

        # Fit road plane on the points of this segment by RANSAC 
        pcl_data = pcl.PointCloud(Points_of_seg_id_XYZ.astype(np.float32))
        seg = pcl_data.make_segmenter() 
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # max distance for a point to be considered fitting the model 
        seg.set_distance_threshold(_seg_distance_thres) 
        seg.set_MaxIterations(800)  # in the original paper, it is set to 800
        seg.set_optimize_coefficients(True) 

        # below takes too much time 
        # indices correspond to points on surface (ax + by + cz + d = 0)
        # coefficients: a = coefficients[0], b = coefficients[1], c = coefficients[2], d = coefficients[3]
        plane_points_indices, coefficients = seg.segment()

        plane_points_mask = np.zeros((Points_of_seg_id.shape[0]), dtype=bool)
        plane_points_mask[plane_points_indices] = True 
        plane_points = Points_of_seg_id[plane_points_mask] 
        
        # Extract points that donot fit the RANSAC plane model
        non_plane_points_mask = np.logical_not(plane_points_mask)
        non_plane_points = Points_of_seg_id[non_plane_points_mask]

        Ground_Points.append(plane_points) 
        NoGround_Points.append(non_plane_points)

    Ground_Points = np.vstack(Ground_Points)
    NoGround_Points = np.vstack(NoGround_Points)

    return Ground_Points, NoGround_Points
    
def compute_FeaturePoint(Ground_Points_Arrange, ScanID_in_range, NumScans): 
        _AngularRes = 0.16
        angleRegionThres=_AngularRes*4

        # compute ori 
        ori_GroundPointArrange = np.arctan2(Ground_Points_Arrange[:, 1], Ground_Points_Arrange[:, 0])*180/math.pi  
        mask_lt_zero = ori_GroundPointArrange < 0 
        ori_GroundPointArrange[mask_lt_zero] += 360 
        
        # 提取高度特征点
        _HeightRegion = 5
        _HeightMaxThres = 0.5
        _HeightMinThres = 0.02 
        _HeightSigmaThres = 0.01 
        _HeightPointsIndex = [] 

        # 提取平滑特征点
        _CurvatureRegion = 5 
        _CurvatureThres = 0.001
        pointWeight = -2 * _CurvatureRegion # we will sum in range, which will include point kkk one more time
        _CurvaturePointsIndex = []  

        # 平面距离特征点
        _DistanceHorizonThres = 2.0 
        distanceHorizonThresholds = np.linalg.norm(Ground_Points_Arrange[:,:2], axis=1)*math.pi*_AngularRes/180
        distancePres = np.linalg.norm(Ground_Points_Arrange[1:,:2] - Ground_Points_Arrange[:-1,:2], axis=1) 
        _DistanceHorizonPointsIndex = []

        # 垂直距离特征
        # _DistanceVerticalThres = 0.0 
        # _DistanceVerticlePointsIndex = []
        # distance_xy = np.linalg.norm(Ground_Points_Arrange[:, :2], axis=1)
        # angleVerticals = np.arctan(Ground_Points_Arrange[:, 2] / distance_xy)
        # distancePreZs=np.abs(Ground_Points_Arrange[1:,2]-Ground_Points_Arrange[:-1,2])
        if True: 
            for kk in range(NumScans):
                start_kscan = ScanID_in_range[kk][0]
                end_kscan = ScanID_in_range[kk][1]
                # 跳过点数小于100的scan
                if end_kscan < start_kscan + 101: 
                    continue 
                
                # STEP 1
                # 提取高度特征点
                for kkk1 in range(start_kscan+_HeightRegion, end_kscan - _HeightRegion + 1): # kkk can be end_kscan - _HeightRegion 
                    # print(kkk1)
                    # for points whoes idx within [start_kscan+_HeightRegion, end_kscan - _HeightRegion]
                    # compute horizon difference 
                    ori_1 = ori_GroundPointArrange[kkk1-_HeightRegion]
                    ori_2 = ori_GroundPointArrange[kkk1+_HeightRegion]
                    abs_ori = abs(ori_1 - ori_2) 
                    if abs_ori > _HeightRegion*2*angleRegionThres: 
                        continue 

                    # 绝对高度点
                    region_max_z = max(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2])  
                    region_min_z = min(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2]) 
                    heightDiff=region_max_z - region_min_z
                    
                    # 高度方差点
                    # standart deviation (z)
                    sigma_height = np.std(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2]) 
                    # 方差和绝对高度都满足条件视为候选curb点
                    if heightDiff >= _HeightMinThres and _HeightMaxThres >= heightDiff and sigma_height > _HeightSigmaThres: 
                        _HeightPointsIndex.append(kkk1)

                # STEP 2
                # 提取平滑特征点
                # note!!!!: we should fuse the below with the above height feature point together to 
                # optimize the whole procedure!!!
                for kkk2 in range(start_kscan+_CurvatureRegion, end_kscan - _CurvatureRegion + 1):
                    ori_1 = ori_GroundPointArrange[kkk2-_CurvatureRegion]
                    ori_2 = ori_GroundPointArrange[kkk2+_CurvatureRegion]
                    abs_ori = abs(ori_1 - ori_2) 

                    if abs_ori > _CurvatureRegion*2*angleRegionThres:
                        continue
                    
                    # original method 
                    diffX = pointWeight *  Ground_Points_Arrange[kkk2,0]
                    diffY = pointWeight *  Ground_Points_Arrange[kkk2,1] 
                    diffZ = pointWeight *  Ground_Points_Arrange[kkk2,2]
                    for j in range(1, _CurvatureRegion+1): 
                        diffX += Ground_Points_Arrange[kkk2+j,0] + Ground_Points_Arrange[kkk2-j,0] 
                        diffY += Ground_Points_Arrange[kkk2+j,1] + Ground_Points_Arrange[kkk2-j,1] 
                        diffZ += Ground_Points_Arrange[kkk2+j,2] + Ground_Points_Arrange[kkk2-j,2] 
                    curvatureValue = sqrt(diffX*diffX + diffY * diffY + diffZ*diffZ) / _CurvatureRegion / sqrt(Ground_Points_Arrange[kkk2, 0]*Ground_Points_Arrange[kkk2, 0] + \
                    Ground_Points_Arrange[kkk2, 1]*Ground_Points_Arrange[kkk2, 1] + Ground_Points_Arrange[kkk2, 2]*Ground_Points_Arrange[kkk2, 2])
                        
                    # my new way (below has problem)
                    # [0]:X, [1]: Y, [2]: Z  
                    # diffXYZ = pointWeight * Ground_Points_Arrange[kkk2,:3] 
                    # 11 x 3 
                    # diffXYZ += np.sum(Ground_Points_Arrange[kkk2-_CurvatureRegion:kkk2+_CurvatureRegion+1, :3], axis=0)
                    # curvatureValue = np.linalg.norm(diffXYZ) /(_CurvatureRegion*np.linalg.norm(Ground_Points_Arrange[kkk2]))

                    if curvatureValue>_CurvatureThres: 
                        _CurvaturePointsIndex.append(kkk2)

                # 平面距离特征点
                for kkk in range(start_kscan+1, end_kscan):
                    ori_1 = ori_GroundPointArrange[kkk-1]
                    ori_2 = ori_GroundPointArrange[kkk+1]
                    abs_ori = abs(ori_1 - ori_2) 

                    if abs_ori > 2 *angleRegionThres:
                        continue
                    
                    # x,y
                    distanceHorizonThreshold = distanceHorizonThresholds[kkk]
                    distancePre = distancePres[kkk-1]  # note kkk -1  
                    if distancePre<distanceHorizonThreshold*_DistanceHorizonThres:
                        _DistanceHorizonPointsIndex.append(kkk) 

                # 垂直距离特征
                # currently we donot use this feature 
                # for kkk in range(start_kscan+1, end_kscan):
                #     ori_1 = Ground_Points_Arrange[kkk-1]
                #     ori_2 =  Ground_Points_Arrange[kkk-1]
                #     abs_ori = abs(ori_1 - ori_2) 

                #     if abs_ori > 2 * angleRegionThres:
                #         continue

                #     angleVertical = angleVerticals[kkk]
                #     distanceHorizonThreshold=abs(math.sin(angleVertical))*distance_xy[kkk]*math.pi*_AngularRes/180
                #     distancePreZ=distancePreZs[kkk-1]
                #     
                #     if distancePreZ> distanceHorizonThreshold*_DistanceVerticalThres:
                #         _DistanceVerticlePointsIndex.append(kkk) 
    
        # print(" height index number is {0}".format(len(_HeightPointsIndex)))
        # print(" curvature index number is {0}".format(len(_CurvaturePointsIndex)))
        # print(" distance horizontal index number is {0}".format(len(_DistanceHorizonPointsIndex)))
        # print(" distance vertical index number is {0}".format(len(_DistanceVerticlePointsIndex)))

        # we only use horitonal
        Ground_Filtered_IDS = reduce(np.intersect1d, (_HeightPointsIndex, _CurvaturePointsIndex, _DistanceHorizonPointsIndex))
        Feature_Points = Ground_Points_Arrange[Ground_Filtered_IDS]
        # print("Final feature point number is {0}".format(len(Ground_Filtered_IDS)))

        return Feature_Points 
     

@njit(nopython=True)
def compute_FeaturePoint_fast(Ground_Points_Arrange, ori_GroundPointArrange, distanceHorizonThresholds, distancePres, ScanID_in_range, NumScans): 
        _AngularRes = 0.16
        angleRegionThres=_AngularRes*4

        # compute ori 
        # 提取高度特征点
        _HeightRegion = 5
        _HeightMaxThres = 0.5
        _HeightMinThres = 0.02 
        _HeightSigmaThres = 0.01 
        _HeightPointsIndex = [] 

        # 提取平滑特征点
        _CurvatureRegion = 5 
        _CurvatureThres = 0.001
        pointWeight = -2 * _CurvatureRegion # we will sum in range, which will include point kkk one more time
        _CurvaturePointsIndex = []  

        # 平面距离特征点
        _DistanceHorizonThres = 2.0 
        _DistanceHorizonPointsIndex = []

        # 垂直距离特征
        # _DistanceVerticalThres = 0.0 
        # _DistanceVerticlePointsIndex = []
        # distance_xy = np.linalg.norm(Ground_Points_Arrange[:, :2], axis=1)
        # angleVerticals = np.arctan(Ground_Points_Arrange[:, 2] / distance_xy)
        # distancePreZs=np.abs(Ground_Points_Arrange[1:,2]-Ground_Points_Arrange[:-1,2])
    
        for kk in range(NumScans):
            start_kscan = ScanID_in_range[kk][0]
            end_kscan = ScanID_in_range[kk][1]
            # 跳过点数小于100的scan
            if end_kscan < start_kscan + 101: 
                continue 
            
            # STEP 1
            # 提取高度特征点
            for kkk1 in range(start_kscan+_HeightRegion, end_kscan - _HeightRegion + 1): # kkk can be end_kscan - _HeightRegion 
                # print(kkk1)
                # for points whoes idx within [start_kscan+_HeightRegion, end_kscan - _HeightRegion]
                # compute horizon difference 
                ori_1 = ori_GroundPointArrange[kkk1-_HeightRegion]
                ori_2 = ori_GroundPointArrange[kkk1+_HeightRegion]
                abs_ori = abs(ori_1 - ori_2) 
                if abs_ori > _HeightRegion*2*angleRegionThres: 
                    continue 

                # 绝对高度点
                region_max_z = max(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2])  
                region_min_z = min(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2]) 
                heightDiff=region_max_z - region_min_z
                
                # 高度方差点
                # standart deviation (z)
                sigma_height = np.std(Ground_Points_Arrange[kkk1-_HeightRegion:kkk1+_HeightRegion+1, 2]) 
                # 方差和绝对高度都满足条件视为候选curb点
                if heightDiff >= _HeightMinThres and _HeightMaxThres >= heightDiff and sigma_height > _HeightSigmaThres: 
                    _HeightPointsIndex.append(kkk1)

            # STEP 2
            # 提取平滑特征点
            # note!!!!: we should fuse the below with the above height feature point together to 
            # optimize the whole procedure!!!
            for kkk2 in range(start_kscan+_CurvatureRegion, end_kscan - _CurvatureRegion + 1):
                ori_1 = ori_GroundPointArrange[kkk2-_CurvatureRegion]
                ori_2 = ori_GroundPointArrange[kkk2+_CurvatureRegion]
                abs_ori = abs(ori_1 - ori_2) 

                if abs_ori > _CurvatureRegion*2*angleRegionThres:
                    continue
                
                # original method 
                diffX = pointWeight *  Ground_Points_Arrange[kkk2,0]
                diffY = pointWeight *  Ground_Points_Arrange[kkk2,1] 
                diffZ = pointWeight *  Ground_Points_Arrange[kkk2,2]
                for j in range(1, _CurvatureRegion+1): 
                    diffX += Ground_Points_Arrange[kkk2+j,0] + Ground_Points_Arrange[kkk2-j,0] 
                    diffY += Ground_Points_Arrange[kkk2+j,1] + Ground_Points_Arrange[kkk2-j,1] 
                    diffZ += Ground_Points_Arrange[kkk2+j,2] + Ground_Points_Arrange[kkk2-j,2] 
                curvatureValue = sqrt(diffX*diffX + diffY * diffY + diffZ*diffZ) / _CurvatureRegion / sqrt(Ground_Points_Arrange[kkk2, 0]*Ground_Points_Arrange[kkk2, 0] + \
                Ground_Points_Arrange[kkk2, 1]*Ground_Points_Arrange[kkk2, 1] + Ground_Points_Arrange[kkk2, 2]*Ground_Points_Arrange[kkk2, 2])
                    
                # my new way (below has problem)
                # [0]:X, [1]: Y, [2]: Z  
                # diffXYZ = pointWeight * Ground_Points_Arrange[kkk2,:3] 
                # 11 x 3 
                # diffXYZ += np.sum(Ground_Points_Arrange[kkk2-_CurvatureRegion:kkk2+_CurvatureRegion+1, :3], axis=0)
                # curvatureValue = np.linalg.norm(diffXYZ) /(_CurvatureRegion*np.linalg.norm(Ground_Points_Arrange[kkk2]))

                if curvatureValue>_CurvatureThres: 
                    _CurvaturePointsIndex.append(kkk2)

            # 平面距离特征点
            for kkk in range(start_kscan+1, end_kscan):
                ori_1 = ori_GroundPointArrange[kkk-1]
                ori_2 = ori_GroundPointArrange[kkk+1]
                abs_ori = abs(ori_1 - ori_2) 

                if abs_ori > 2 *angleRegionThres:
                    continue
                
                # x,y
                distanceHorizonThreshold = distanceHorizonThresholds[kkk]
                distancePre = distancePres[kkk-1]  # note kkk -1  
                if distancePre<distanceHorizonThreshold*_DistanceHorizonThres:
                    _DistanceHorizonPointsIndex.append(kkk) 

                # 垂直距离特征
                # currently we donot use this feature 
                # for kkk in range(start_kscan+1, end_kscan):
                #     ori_1 = Ground_Points_Arrange[kkk-1]
                #     ori_2 =  Ground_Points_Arrange[kkk-1]
                #     abs_ori = abs(ori_1 - ori_2) 

                #     if abs_ori > 2 * angleRegionThres:
                #         continue

                #     angleVertical = angleVerticals[kkk]
                #     distanceHorizonThreshold=abs(math.sin(angleVertical))*distance_xy[kkk]*math.pi*_AngularRes/180
                #     distancePreZ=distancePreZs[kkk-1]
                #     
                #     if distancePreZ> distanceHorizonThreshold*_DistanceVerticalThres:
                #         _DistanceVerticlePointsIndex.append(kkk) 
    
        # print(" height index number is {0}".format(len(_HeightPointsIndex)))
        # print(" curvature index number is {0}".format(len(_CurvaturePointsIndex)))
        # print(" distance horizontal index number is {0}".format(len(_DistanceHorizonPointsIndex)))
        # print(" distance vertical index number is {0}".format(len(_DistanceVerticlePointsIndex)))

        return _HeightPointsIndex, _CurvaturePointsIndex, _DistanceHorizonPointsIndex

def compute_segmentation_angle(filtered_noground_points, filtered_noground_points_distance):
    resolution = 1 # resolution of the polar grid 
    ori_no_ground = np.arctan2(filtered_noground_points[:, 1], filtered_noground_points[:, 0]) * 180 / math.pi 
    ori_lt_zero = ori_no_ground < 0 
    ori_no_ground[ori_lt_zero] += 360 

    # Beam Band Model 
    # generate polar grid, centered at the Lidar position
    # each grid cell contains points from ground_point_no 
    _grid_map_vec = [[] for _ in range(360//resolution)]
    _distance_vec = [[] for _ in range(360//resolution)]
    
    for k in range(filtered_noground_points.shape[0]):
        # degrees 1, 2, 3, ..., 360
        # which polar grid does this point belong to 
        segment_index = (int)(ori_no_ground[k] / resolution)

        if segment_index < 360:     
            # original 
            # _grid_map_vec[segment_index].append(filtered_noground_points[k])
            # now let us just append the distance 
            _grid_map_vec[segment_index].append(filtered_noground_points_distance[k])

    # compute the beam length and then preform median filtering
    for k in range(360//resolution):
        # sorts points_in_kgrid based on its distance
        points_distance_in_kgrid = _grid_map_vec[k]
        points_distance_in_kgrid.sort()
        
        if len(points_distance_in_kgrid) > 0:
            # compute beam length 
            distance_min = points_distance_in_kgrid[0]
            distance_max = points_distance_in_kgrid[-1]  
            _distance_vec[k] = (distance_min / distance_max, k)
        else: # doesnot contain any point 
            _distance_vec[k] = (1.0, k) 
    
    # apply median filter to the distance 
    _distance_vec_front = [] 
    _distance_vec_rear = [] 
    # window size is 6, takes 3 former elements, 3 latter elements 
    _distance_vec_front.append(_distance_vec[0]) 
    _distance_vec_front.append(_distance_vec[1]) 
    _distance_vec_front.append(_distance_vec[2]) 
    _distance_vec_front.append(_distance_vec[359]) 
    _distance_vec_front.append(_distance_vec[358]) 
    _distance_vec_front.append(_distance_vec[357]) 
    for k in range(3, 360//resolution-3): 
        temp = [ele[0] for ele in _distance_vec[k-3:k+4]] # take out distances of element k-3, k-2, k-1, k, k+1, k+2, k+3 
        temp.sort()  

        if k > 90 and k < 270:  # rear region 
            _distance_vec_rear.append((temp[3], k))  
        else: # front region 
            _distance_vec_front.append((temp[3], k))

    # compute the two dominant beam which represents the direction ofroad. 
    # one beam angle is for the front road, the other beam angle is for the rear road.  
    # sorting _distance_vec_front
    _distance_vec_front.sort(reverse=True, key=lambda x: x[0]) # sort based on beam length 
    # sorting _distance_vec_rear 
    _distance_vec_rear.sort(reverse=True, key=lambda x: x[0])

    # compute segmentation angle for the car front region
    # note: the first is the longest beam 
    _segmentAngle = [-1, -1] 
    if _distance_vec_front[0][0] == 1.0: 
        max_distance_angle_left = [] 
        max_distance_angle_right = [] 
        for k in range(len(_distance_vec_front)): 
            # 0-90 
            if _distance_vec_front[k][0] == 1.0 and  _distance_vec_front[k][1] < 90:
                max_distance_angle_left.append(_distance_vec_front[k][1]) 

            # 270-360 
            if _distance_vec_front[k][0] == 1.0 and  _distance_vec_front[k][1] > 270:
                max_distance_angle_right.append(_distance_vec_front[k][1]) 

        # sort 
        max_distance_angle_right.sort(reverse=True) 
        max_distance_angle_left.sort(reverse=True)
        # insert max_distance_angle_right to the end of max_distance_angle_left
        max_distance_angle_left = max_distance_angle_left + max_distance_angle_right
        _segmentAngle[0] = max_distance_angle_left[int(len(max_distance_angle_left) / 2)]
    else: # the longest beam length is not 1.0
        _segmentAngle[0] = _distance_vec_front[0][1]

    # compute the segmentation angle of car rear 
    if _distance_vec_rear[0][0] == 1.0: 
        max_distance_angle = [] 
        for k in range(len(_distance_vec_rear)):
            if _distance_vec_rear[k][0] == 1:
                max_distance_angle.append(_distance_vec_rear[k][1]) 
        max_distance_angle.sort() 
        _segmentAngle[1] = max_distance_angle[int(len(max_distance_angle) / 2)]
    else: 
        _segmentAngle[1] = _distance_vec_rear[0][1]

    return _segmentAngle

## fast version 
def compute_segmentation_angle_fast(filtered_noground_points, ori_no_ground, filtered_noground_points_distance):
    # resolution = 1 # resolution of the polar grid 

    # Beam Band Model 
    # generate polar grid, centered at the Lidar position
    # each grid cell contains points from ground_point_no 
    ori_lt360_mask = ori_no_ground < 360 
    ori_no_ground = ori_no_ground[ori_lt360_mask]
    filtered_noground_points = filtered_noground_points[ori_lt360_mask]
    filtered_noground_points_distance = filtered_noground_points_distance[ori_lt360_mask]

    # compute the beam length and then preform median filtering
    _distance_vec = [(1.0, k) for k in range(360)] # by default, the beam length is (1.0, k)  
    
    for k in range(360):
        points_distance_in_kgrid = filtered_noground_points_distance[ori_no_ground == k]        
        if len(points_distance_in_kgrid) > 0:
            # compute beam length
            distance_min = np.min(points_distance_in_kgrid)
            distance_max = np.max(points_distance_in_kgrid)
            _distance_vec[k] = (distance_min / distance_max, k)

    # apply median filter to the distance 
    _distance_vec_front = [] 
    _distance_vec_rear = [] 
    # window size is 6, takes 3 former elements, 3 latter elements 
    _distance_vec_front.append(_distance_vec[0]) 
    _distance_vec_front.append(_distance_vec[1]) 
    _distance_vec_front.append(_distance_vec[2]) 
    _distance_vec_front.append(_distance_vec[359]) 
    _distance_vec_front.append(_distance_vec[358]) 
    _distance_vec_front.append(_distance_vec[357]) 
    for k in range(3, 357): 
        temp = [ele[0] for ele in _distance_vec[k-3:k+4]] # take out distances of element k-3, k-2, k-1, k, k+1, k+2, k+3 
        temp.sort()  

        if k > 90 and k < 270:  # rear region 
            _distance_vec_rear.append((temp[3], k))  
        else: # front region 
            _distance_vec_front.append((temp[3], k))

    # compute the two dominant beam which represents the direction ofroad. 
    # one beam angle is for the front road, the other beam angle is for the rear road.  
    # sorting _distance_vec_front
    _distance_vec_front.sort(reverse=True, key=lambda x: x[0]) # sort based on beam length 
    # sorting _distance_vec_rear 
    _distance_vec_rear.sort(reverse=True, key=lambda x: x[0])

    # compute segmentation angle for the car front region
    # note: the first is the longest beam 
    _segmentAngle = [-1, -1] 
    if _distance_vec_front[0][0] == 1.0: 
        max_distance_angle_left = [] 
        max_distance_angle_right = [] 
        for k in range(len(_distance_vec_front)): 
            # 0-90 
            if _distance_vec_front[k][0] == 1.0 and  _distance_vec_front[k][1] < 90:
                max_distance_angle_left.append(_distance_vec_front[k][1]) 

            # 270-360 
            if _distance_vec_front[k][0] == 1.0 and  _distance_vec_front[k][1] > 270:
                max_distance_angle_right.append(_distance_vec_front[k][1]) 

        # sort 
        max_distance_angle_right.sort(reverse=True) 
        max_distance_angle_left.sort(reverse=True)
        # insert max_distance_angle_right to the end of max_distance_angle_left
        max_distance_angle_left = max_distance_angle_left + max_distance_angle_right
        _segmentAngle[0] = max_distance_angle_left[int(len(max_distance_angle_left) / 2)]
    else: # the longest beam length is not 1.0
        _segmentAngle[0] = _distance_vec_front[0][1]

    # compute the segmentation angle of car rear 
    if _distance_vec_rear[0][0] == 1.0: 
        max_distance_angle = [] 
        for k in range(len(_distance_vec_rear)):
            if _distance_vec_rear[k][0] == 1:
                max_distance_angle.append(_distance_vec_rear[k][1]) 
        max_distance_angle.sort() 
        _segmentAngle[1] = max_distance_angle[int(len(max_distance_angle) / 2)]
    else: 
        _segmentAngle[1] = _distance_vec_rear[0][1]

    return _segmentAngle

def classify_feature_points(Feature_Points, _segmentAngle):
    # based on filtered_noground_points, we classify right and left lane 
    clusterPtrLR = [[], []] # 0: left lane, [1] is right lane 

    for k in range(Feature_Points.shape[0]): # Feature_Points: numpy array of N x 4 
        # front
        if Feature_Points[k][0] > 0:
            if Feature_Points[k][1] > Feature_Points[k][0] * np.tan(_segmentAngle[0]*math.pi/180):
                # left feature points  
                clusterPtrLR[0].append(Feature_Points[k])
            else: 
                # right feature points 
                clusterPtrLR[1].append(Feature_Points[k])
        else: # rear 
            if Feature_Points[k][1] > Feature_Points[k][0] * np.tan(_segmentAngle[1]*math.pi/180): 
                # left feature points 
                clusterPtrLR[0].append(Feature_Points[k])
            else: 
                # right feature points 
                clusterPtrLR[1].append(Feature_Points[k])
    
    return clusterPtrLR

def classify_feature_points_fast(Feature_Points, _segmentAngle):
    #
    # based on filtered_noground_points, we classify right and left lane 
    clusterPtrLR = [[], []] # 0: left lane, [1] is right lane 
    
    mask_front = Feature_Points[:, 0] > 0 
    front_feature_points = Feature_Points[mask_front] 
    # front left 
    front_left_mask = front_feature_points[:, 1] > front_feature_points[:, 0] * np.tan(_segmentAngle[0]*math.pi/180) 
    front_left = front_feature_points[front_left_mask]
    # front right 
    front_right_mask = np.logical_not(front_left_mask) 
    front_right = front_feature_points[front_right_mask]

    mask_rear = np.logical_not(mask_front) 
    rear_feature_points = Feature_Points[mask_rear] 
    # rear right 
    rear_left_mask = rear_feature_points[:, 1] > rear_feature_points[:, 0] * np.tan(_segmentAngle[1]*math.pi/180)
    rear_left = rear_feature_points[rear_left_mask]  
    # rear right 
    rear_right_mask = np.logical_not(rear_left_mask)
    rear_right = rear_feature_points[rear_right_mask] 
    
    left_lane = np.vstack([front_left, rear_left])
    right_lane = np.vstack([front_right, rear_right]) 
    clusterPtrLR = [left_lane, right_lane] 

    return clusterPtrLR


def distanceFilterByCartesianGrid(clusterPtrLR_data, left=True):
    _gridRes = 0.5 
    _gridNum = 200 
    ########################################################################
    # generate cartesian grid 
    ########################################################################
    grid_map_vec_carte = [[] for _ in range(_gridNum)] 
    for k, v in enumerate(clusterPtrLR_data):
        row = int(v[0]/_gridRes + _gridNum/2) 
        if row >=0 and row < _gridNum:
            # store the feature into the grid they belong
            grid_map_vec_carte[row].append(v)
                        
    # For each grid, we select the nearest point as the candidate point 
    pointcloud_distancefiltered = [] 
    for k in range(len(grid_map_vec_carte)): 
        if len(grid_map_vec_carte[k]) > 0:
            if left: # for left, we choose point with y of minimum value 
                xyzi_min_id = np.argmin(grid_map_vec_carte[k], axis=0)
                y_id = xyzi_min_id[1] 
            else: # for right, we choose point with y of maximum value 
                xyzi_max_id = np.argmax(grid_map_vec_carte[k], axis=0)
                y_id = xyzi_max_id[1] 

            pointcloud_distancefiltered.append(grid_map_vec_carte[k][y_id])
        
    # set intensity to its id 
    for k, v in enumerate(pointcloud_distancefiltered):
        v[3] = k 

    return pointcloud_distancefiltered

def distanceFilterByCartesianGrid_fast(clusterPtrLR_data, left=True):
    _gridRes = 0.5 
    _gridNum = 200 
    ########################################################################
    # generate cartesian grid 
    ########################################################################
    start = time.time() 
    grid_map_vec_carte = [[] for _ in range(_gridNum)] 
    row_IDs = clusterPtrLR_data[:, 0] / _gridRes + _gridNum/2
    row_IDs = row_IDs.astype(int)
    for k, v in enumerate(clusterPtrLR_data):
        row = row_IDs[k]
        if row >=0 and row < _gridNum:
            # store the feature into the grid they belong
            grid_map_vec_carte[row].append(v)
                        
    # For each grid, we select the nearest point as the candidate point 
    print('within grid fast takes {0} ms.'.format((time.time()-start)*1000))

    pointcloud_distancefiltered = [] 
    for k in range(len(grid_map_vec_carte)): 
        if len(grid_map_vec_carte[k]) > 0:
            if left: # for left, we choose point with y of minimum value 
                xyzi_min_id = np.argmin(grid_map_vec_carte[k], axis=0)
                y_id = xyzi_min_id[1] 
            else: # for right, we choose point with y of maximum value 
                xyzi_max_id = np.argmax(grid_map_vec_carte[k], axis=0)
                y_id = xyzi_max_id[1] 

            pointcloud_distancefiltered.append(grid_map_vec_carte[k][y_id])

    # set intensity to its id 
    for k, v in enumerate(pointcloud_distancefiltered):
        v[3] = k 

    return pointcloud_distancefiltered

def ransac_curve(pointcloud_distancefiltered, _curveFitThres):
    nData = pointcloud_distancefiltered.shape[0]
    xs = pointcloud_distancefiltered[:, 0]
    ys = pointcloud_distancefiltered[:, 1]

    # Matrix A 
    A = np.random.rand(nData, 3)
    B = np.random.rand(nData, 1) 
    # set its value 
    A[:, 0] = xs**2 
    A[:, 1] = xs 
    A[:, 2] = 1.0 
    B[:, 0] = ys 
    # RANSAC fiting 
    N = 300 # iterations 
    residualThreshold = _curveFitThres # residual threshold for fiting the curve 
    n_sample = 3 
    max_cnt = 0 
    coefficients = np.random.rand(3, 1) # model for fiting 
    for k in range(N): 
        # random sampling three points whose ids are k_id[0], k_id[1], k_id[2]
        k_id = np.random.choice(nData, 3, replace=False)

        # random sample is k[0], k[1], k[2] 
        # mode lestimation 
        AA = np.random.rand(3, 3)
        BB = np.random.rand(3, 1) 

        # set its values 
        AA[:, 0] = xs[k_id]**2 
        AA[:, 1] = xs[k_id] 
        AA[:, 2] = 1.0 
        BB[:,0] = ys[k_id]

        # compute the inverse of AA 
        AA_inv = np.linalg.inv(AA) # 3 x 3  
        X = np.matmul(AA_inv, BB) # 3 x 1
        
        # evaluation  
        if X[0][0] < 0.1:
            residual = abs(B - np.matmul(A, X))  
            cnt = 0 # number of inliers 
            mask_inliers = residual[:, 0] < residualThreshold 
            cnt = np.sum(mask_inliers) 
            if cnt > max_cnt: 
                coefficients = X 
                max_cnt = cnt 
        
    
    # optional LS fiting 
    residual = abs(np.matmul(A, coefficients) - B) 
    mask_inlier_ind = residual < residualThreshold
    mask_inlier_ind = mask_inlier_ind.reshape(-1)

    x_inliers = xs[mask_inlier_ind]  
    A2 = np.random.rand(x_inliers.shape[0], 3) 
    B2 = np.random.rand(x_inliers.shape[0], 1) 
    A2[:, 0] = x_inliers ** 2 
    A2[:, 1] = x_inliers 
    A2[:, 2] = 1.0 
    B2[:, 0] = ys[mask_inlier_ind] 

    A2_inv = np.linalg.pinv(A2) 
    X = np.matmul(A2_inv, B2) 
    residual_opt = abs(np.matmul(A, X) - B)
    vect_index_opt_mask = residual_opt < residualThreshold
    vect_index_opt_mask = vect_index_opt_mask.reshape(-1)
    cloud_initial = pointcloud_distancefiltered[vect_index_opt_mask]
    print("curb line model coefficient is {0} {1} {2}".format(X[0], X[1], X[2])) 

    return cloud_initial # could be left/right point cloud initial 

# Converter 
class Lidar_LaneDet_Manager():
    def __init__(self):
        self.lidar_topic = rospy.get_param('~lidar_topic', '/kitti/velo/pointcloud')
        # Define subscribers
        self.pc_lidar_sub = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size = 1, buff_size = 2**24)

        # Define publishers
        rospy.loginfo("Launched node for Lidar based Lane Det.")
        # Load publisher topics
        self.left_lane_topic = rospy.get_param('~left_lane_topic', '/left_lane_boundary')
        self.right_lane_topic = rospy.get_param('~right_lane_topic', '/right_lane_boundary')
        self.pub_left_lane = rospy.Publisher(self.left_lane_topic, PointCloud2, queue_size=10)
        self.pub_right_lane = rospy.Publisher(self.right_lane_topic, PointCloud2, queue_size=10)

        # Spin
        rospy.spin()

    def lidar_callback(self, msg):
        start = time.time() 
        # related data info 
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg) # msg is of type pointcloud2 
        points_np_array = get_xyzi_points(msg_cloud, True) # size: N x 4 (last row is (0, 0, 0, 1)), so it will maintrain its intensity, format: (x, y, z, intensity)
        print("points shape: {0}".format(points_np_array.shape))
        # points_np_array = band_pass_points(points_np_array)
        
        # just for debugging, let us readin the file 
        debug = True 
        if debug: 
            points_np_array = np.fromfile("/home/ps/hxw_projects/Hongfeng_LaneDet/catkin_ws/src/lane_detection_lidar/data/000428.bin", dtype=np.float32).reshape(-1, 4) 
        ###############################################################################################################
        # STEP 1
        # reorganize the point cloud according to the characterstics of point cloud data from kitti dataset, and store the laserID
        # the intensity value for each point 
        # process by ori 
        ###############################################################################################################
        # start = time.time() 
        # 10 ms 
        Points_By_Scan_ID = organize_kitti_by_scanIDs_fast(points_np_array) 
        # if True: # check whether faster version is correct 
        #   Points_By_Scan_ID = organize_kitti_by_scanIDs(points_np_array) # N' x 4 which are arranged by the order of the scan
        #   Points_By_Scan_ID2 = organize_kitti_by_scanIDs_fast(points_np_array) 
        #   # check whether they are equal 
        #   print(np.all(Points_By_Scan_ID == Points_By_Scan_ID2))  
        # print('takes {0} ms.'.format((time.time()-start)*1000))
        if False: # added by me to decrease the point cloud to further decrease the computation 
            Points_By_Scan_ID = band_pass_points(Points_By_Scan_ID)

        print("final arranged raw point number is: {0}".format(Points_By_Scan_ID.shape[0]))

        ################################################################################################################
        # STEP 2 
        # ground/non-ground point extraction 
        ################################################################################################################
        # v1 
        # start = time.time() 
        Ground_Points, NoGround_Points  = extract_ground_non_ground(Points_By_Scan_ID)
        # print('takes {0} ms.'.format((time.time()-start)*1000))
        print("ground points number: {0}".format(Ground_Points.shape[0]))
        print("non-ground points number: {0}".format(NoGround_Points.shape[0]))
        
        # start = time.time() 
        Point_Segs_By_X = [] 
        # 0 =< x =< 15 , |y| < 30 
        Point_Seg_By_X0 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= 0, 15 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
        Point_Segs_By_X.append(Point_Seg_By_X0) 
        # -15 =< x < 0 , |y| < 30
        Point_Seg_By_X1 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -15, Points_By_Scan_ID[:,0] < 0, abs(Points_By_Scan_ID[:,1]) < 30))] 
        Point_Segs_By_X.append(Point_Seg_By_X1) 
        # (15, 30] and [-30, -15)
        # 15 < x =< 30 , |y| < 30 
        Point_Seg_By_X2 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] > 15, 30 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
        Point_Segs_By_X.append(Point_Seg_By_X2) 
        # -30 =< x < -15 , |y| < 30
        Point_Seg_By_X3 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -30, Points_By_Scan_ID[:,0] < -15, abs(Points_By_Scan_ID[:,1]) < 30))] 
        Point_Segs_By_X.append(Point_Seg_By_X3) 
        # (30, 40] and [-40, -30)
        # 30 < x =< 40 , |y| < 30 
        # Point_Seg_By_X4 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] > 30, 40 >= Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
        # Point_Seg_Dict[4] = Point_Seg_By_X4
        # -40 =< x < -30 , |y| < 30  
        # Point_Seg_By_X5 = Points_By_Scan_ID[np.logical_and.reduce((Points_By_Scan_ID[:,0] >= -40, -30 > Points_By_Scan_ID[:,0], abs(Points_By_Scan_ID[:,1]) < 30))] 
        # Point_Seg_Dict[5] = Point_Seg_By_X5  

        Ground_Points, NoGround_Points  = extract_ground_non_ground_fast(Point_Segs_By_X)
        # print('fast takes {0} ms.'.format((time.time()-start)*1000))
        # Based on ground points 
        # reorganize points by the scanID 
        # Based on intensity (i.e., the scanID) 
        laserCloudScans = {} 
        ScanIDs = list(set(Ground_Points[:, 3])) 
        ScanIDs.sort() 
        ScanID_in_range = [] 
        start_id = 0 
        Ground_Points_Arrange = [] 
        for _, ScanID in enumerate(ScanIDs):
            laserCloudScans[ScanID] = Ground_Points[Ground_Points[:, 3]==ScanID]

        for _, (kth_ring, kth_scan_pc) in enumerate(laserCloudScans.items()):
            ScanID_in_range.append((start_id, start_id+kth_scan_pc.shape[0]-1)) 
            start_id += kth_scan_pc.shape[0]
            Ground_Points_Arrange.append(kth_scan_pc) 
        
        Ground_Points_Arrange = np.vstack(Ground_Points_Arrange)
        NumScans = len(ScanIDs) 
        ################################################################################################################
        # STEP 3 
        # Extract feature points 
        # Input: Ground_Points_Arrange, ScanID_in_range
        ################################################################################################################
        # takes 30000 ms 
        # start = time.time() 
        # Feature_Points = compute_FeaturePoint(Ground_Points_Arrange, ScanID_in_range, NumScans)
        # print('takes {0} ms.'.format((time.time()-start)*1000))
        
        # only takes 25 ms after 1st iteration 
        # start = time.time() 
        ori_GroundPointArrange = np.arctan2(Ground_Points_Arrange[:, 1], Ground_Points_Arrange[:, 0])*180/math.pi  
        mask_lt_zero = ori_GroundPointArrange < 0 
        ori_GroundPointArrange[mask_lt_zero] += 360 
        _AngularRes = 0.16
        distanceHorizonThresholds = np.linalg.norm(Ground_Points_Arrange[:,:2], axis=1)*math.pi*_AngularRes/180
        distancePres = np.linalg.norm(Ground_Points_Arrange[1:,:2] - Ground_Points_Arrange[:-1,:2], axis=1) 

        # to use numba, can not pass in list of list, so let us change it to numpy array 
        ScanID_in_range = np.array(ScanID_in_range) 
        _HeightPointsIndex, _CurvaturePointsIndex, _DistanceHorizonPointsIndex = compute_FeaturePoint_fast(Ground_Points_Arrange, ori_GroundPointArrange, distanceHorizonThresholds, distancePres, ScanID_in_range, NumScans)
        Ground_Filtered_IDS = reduce(np.intersect1d, (_HeightPointsIndex, _CurvaturePointsIndex, _DistanceHorizonPointsIndex))
        Feature_Points = Ground_Points_Arrange[Ground_Filtered_IDS]
        # print('fast takes {0} ms.'.format((time.time()-start)*1000))
        ################################################################################################################
        # STEP 4 
        # based on NoGround_Points and Feature Points, we computer the road boundary points  
        ################################################################################################################
        # removing noisy points if too close (usually the car top) (BEV, x and y)
        NoGround_Points_Distance = np.linalg.norm(NoGround_Points[:,:2], axis=1)
        filterd_mask = NoGround_Points_Distance**2 >=7
        filtered_noground_points = NoGround_Points[filterd_mask]
        filtered_noground_points_distance = NoGround_Points_Distance[filterd_mask]
 
        # compute segmentation angle
        # start = time.time() 
        # _segmentAngle = compute_segmentation_angle(filtered_noground_points, filtered_noground_points_distance)
        # print('takes {0} ms.'.format((time.time()-start)*1000))
        
        # fast version 
        ori_no_ground = np.arctan2(filtered_noground_points[:, 1], filtered_noground_points[:, 0]) * 180 / math.pi 
        ori_lt_zero = ori_no_ground < 0 
        ori_no_ground[ori_lt_zero] += 360 
        ori_no_ground = ori_no_ground.astype(int)
        _segmentAngle = compute_segmentation_angle_fast(filtered_noground_points, ori_no_ground, filtered_noground_points_distance)
        print("The front segmentation angle is {0}".format(_segmentAngle[0])) 
        print("The back segmentation angle is  {0}".format(_segmentAngle[1])) 

        # classify the feature points
        # left feature points (stored in outcloud[0]) and right feature points (stored in outcloud[1]) 
        # based on filtered_noground_points, we classify right and left lane 
        # 35 ms 
        # clusterPtrLR = classify_feature_points(Feature_Points, _segmentAngle) # 0: left lane, [1] is right lane 
        # 0.31 ms 
        clusterPtrLR = classify_feature_points_fast(Feature_Points, _segmentAngle) 
        print("left feature points is  {0}".format(len(clusterPtrLR[0]))) 
        print("right feature points is  {0}".format(len(clusterPtrLR[1])))  
        
        ## all the above is correct 
        # For left Features clusterPtrLR[0], we use distnace filter 
        # start = time.time() 
        # takes 2773.8428115844727 ms 
        pointcloud_distanceleftfiltered = distanceFilterByCartesianGrid_fast(clusterPtrLR[0], left=True)
        pointcloud_distancerightfiltered = distanceFilterByCartesianGrid_fast(clusterPtrLR[1], left=False)

        # set intensity to its id 
        # for k, v in enumerate(pointcloud_distancerightfiltered):
        #     v[3] = k 
        
        # concatenate 
        # below command is useless 
        pointcloud_distancefiltered = pointcloud_distanceleftfiltered + pointcloud_distancerightfiltered

        print("distance filter left (i.e., candidate point) is {0}".format(len(pointcloud_distanceleftfiltered)))
        print("distance filter right (i.e., candiate point) is {0}".format(len(pointcloud_distancerightfiltered)))
    
        ##################################################################
        # Fit a ransac curve 
        ##################################################################
        _curveFitThres = 0.15
        # RANSAC Curve for left road (pointcloud_distanceleftfiltered)
        # convert list of 3D points into numpy array 
        pointcloud_distanceleftfiltered = np.vstack(pointcloud_distanceleftfiltered)
        Leftcloud_initial = ransac_curve(pointcloud_distanceleftfiltered, _curveFitThres)
        # RANSAC curve for right road (pointcloud_distancerightfiltered)
        pointcloud_distancerightfiltered = np.vstack(pointcloud_distancerightfiltered)
        Rightcloud_initial = ransac_curve(pointcloud_distancerightfiltered, _curveFitThres)

        print("before gaussian left point is {0}".format(Leftcloud_initial.shape[0])) 
        print("before gaussian right point is {0}".format(Rightcloud_initial.shape[0]))
        print('all takes {0} ms.'.format((time.time()-start)*1000))
        ########################################################################################
        # In order to improve the robustness of road
        # boundary detection, iterative GPR is further applied to extract
        # road boundary points.
        # for left seed point 
        # GPR has strong outlier rejection abilities
        # Input: P_{candidate,l}, P_{seed,l}
        # based on P_{seed,l}, we filter P_{candidate,l} 
        #########################################################################################
        # currently let us skip it since it doesnot affect the performance too much  
        # in the future we may implement this part 
        # I have made sure the above is correct 
        
        ##########################################################################################
        # Publish the detected road boundary  
        ##########################################################################################
        msg_left_lane_boundary = xyzi_array_to_pointcloud2(Leftcloud_initial, msg)
        msg_right_lane_boundary = xyzi_array_to_pointcloud2(Rightcloud_initial, msg)

        self.pub_left_lane.publish(msg_left_lane_boundary)
        self.pub_right_lane.publish(msg_right_lane_boundary)
 
if __name__=="__main__":
    # Initialize node
    rospy.init_node("Lidar_LaneDet_Manager_Node")
    dm = Lidar_LaneDet_Manager()

