#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy

import numpy as np
import ros_numpy
import colorsys

import message_filters

import tf as tf2
from tf import TransformListener

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

from utils import *
import open3d

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import math
import pandas as pd

pc_stack = np.empty((0, 3), float)
Lane_markers_array = MarkerArray()
Plane_markers_array = MarkerArray()
frame_stack = 10
id_lane_global = 0
id_plane_global = 0
atan_theta_past = 0 


class lane_detection_class:

    def __init__(self):
        self.lidar_pub = rospy.Publisher("/frame_stack", PointCloud2, queue_size=5)
        self.lidar_sub = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.callback, queue_size=5)

        self.pub_Lane_marker = rospy.Publisher('/lane_marker', MarkerArray, queue_size=5)
        self.pub_Plane_marker = rospy.Publisher('/Plane_marker', MarkerArray, queue_size=5)

        self.tf = TransformListener()

    def callback(self, PointCloud2):
        global pc_stack, frame_stack, Lane_markers_array, Plane_markers_array, id_lane_global, id_plane_global, weight_past

        pc_np = get_xyzi_points(pointcloud2_to_array(PointCloud2))
        xyz_points = pc_np[:,:3]
        intensity = pc_np[:,3]
        
        road_pts = extract_points(pc_np)
        odom_mat = get_odom(self.tf,"velo_link", "map")

        if odom_mat is not None:
            points = get_transformation(odom_mat,road_pts)
            pc_stack = np.append(pc_stack, points, axis=0)


        if (PointCloud2.header.seq > 0) and (PointCloud2.header.seq % frame_stack == 0): 
            # print("pub : ", PointCloud2.header.seq)

            height = pc_stack[...,2].mean()-0.5
            
            pc_stack[...,2] = 0
            base_mat = get_odom(self.tf, "map","base_link")
            map_mat = get_odom(self.tf,"base_link", "map")

            if base_mat is not None:
                basepoints = get_transformation(base_mat,pc_stack)

            grid_y = [i for i in basepoints[:,1]]
            
            grid_y_start = min(grid_y)
            grid_y_end = max(grid_y) 
            interval = 1.2

            lane1x = sys.maxsize
            lane1y = sys.maxsize
            
            lane2x = -sys.maxsize - 1
            lane2y = -sys.maxsize - 1
            atan_theta = 0
            len_line = 0
            atan_theta2 = 0
            len_line2 = 0

            while grid_y_start < grid_y_end:
                
                xgrid = []
                ygrid = []
                grid = []

                for j in range(len(grid_y)):
                    if basepoints[:,1][j] >= grid_y_start and basepoints[:,1][j] < grid_y_start + interval:
                        xgrid.append(basepoints[:,0][j])
                        ygrid.append(basepoints[:,1][j])

                if len(ygrid) < 25:
                    grid_y_start = grid_y_start + interval
                    if grid_y_start > grid_y_end:
                        break
                    continue

                temp = pd.DataFrame({'X':xgrid,'Y':ygrid})
                grid = temp.as_matrix()
                ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                                    max_trials=100, 
                                                    min_samples=None,
                                                    residual_threshold=0.05)
                def add_square_feature(X):
                    # X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
                    return X

                sub_cluster_df = grid
                Xpoints = sub_cluster_df[...,0]
                Xpoints = Xpoints.reshape(-1,1)
                Ypoints = sub_cluster_df[...,1]
                Ypoints = Ypoints.reshape(-1,1)
                ransac.fit(add_square_feature(Xpoints), Ypoints)
                inlier_mask = ransac.inlier_mask_

                line_X = np.arange(Xpoints.min(), Xpoints.max())[:, np.newaxis]
                line_y_ransac = ransac.predict(add_square_feature(line_X))

                if len_line2 < (line_X.max()-line_X.min()):
                    atan_theta2 = math.atan((line_y_ransac.max()- line_y_ransac.min()) / (line_X.max()-line_X.min()))
                    len_line2 = line_X.max()-line_X.min()

                line_X = line_X.reshape(-1)
                line_y_ransac = line_y_ransac.reshape(-1)
                line_z = [0 for _ in range(len(line_X))]

                line_tmp = pd.DataFrame({'X':line_X,'Y':line_y_ransac,'Z':line_z})
                line_ransac = line_tmp.as_matrix()
                
    
                if map_mat is not None:
                    line_ransac = get_transformation(map_mat,line_ransac)
                                        
                line_X = line_ransac[:,0].reshape(-1,1)
                line_y_ransac = line_ransac[:,1].reshape(-1,1)

                if line_X.min() < lane1x:
                    lane1x = line_X.min()
                    

                if line_y_ransac.min() < lane1y:
                    lane1y = line_y_ransac.min()
                    
                if line_X.max() > lane2x:
                    lane2x = line_X.max()
                    
                if line_y_ransac.max() > lane2y:
                    lane2y = line_y_ransac.max()
                

                if len_line < (line_X.max()-line_X.min()):
                    atan_theta = math.atan((line_y_ransac.max()- line_y_ransac.min()) / (line_X.max()-line_X.min()))
                    len_line = line_X.max()-line_X.min()

                #print("theta: ",atan_theta2)
                if atan_theta2 < 0.1:


                    quaternion = tf2.transformations.quaternion_from_euler(0,np.pi/2,atan_theta)
                    # if line_theta < 0.3 and line_theta >0.05:
                    Lane_marker = Marker(type=Marker.CYLINDER,
                                        id = id_lane_global,
                                        lifetime=rospy.Duration(300),
                                        pose=Pose(Point(0.0,0.0,0), Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])),
                                        scale=Vector3(0.01, 0.5, line_X.max()-line_X.min()),                           # line width
                                        header=PointCloud2.header,
                                        color=ColorRGBA(1.0, 1.0, 1.0, 1.0)
                                        )


                    

                    Lane_marker.header.frame_id = "/map"


                    l_points = Point()
                    l_points.x = np.median(line_X)
                    l_points.y = np.median(line_y_ransac)
                    l_points.z = height

                    Lane_marker.pose.position = l_points

                    id_lane_global +=1

                    Lane_markers_array.markers.append(Lane_marker)



                    
                grid_y_start = grid_y_start + interval
                if grid_y_start > grid_y_end:
                    break


            rect_point1 = Point(lane1x,lane1y,0) 
            rect_point2 = Point(lane2x,lane2y,0) 
            

            quaternion = tf2.transformations.quaternion_from_euler(0,0,atan_theta)

            Plane_marker = Marker(type=Marker.CUBE, 
                                    header = PointCloud2.header,
                                    action = Marker.ADD,
                                    id = id_plane_global,
                                    scale = Vector3(np.fabs(rect_point1.x - rect_point2.x),
                                                    np.fabs(rect_point1.y - rect_point2.y), 
                                                    np.fabs(rect_point1.z - rect_point2.z)),
                                    color = ColorRGBA(0.0, 0.0, 0.0, 1.0),
                                    pose= Pose(Point((rect_point1.x - rect_point2.x) / 2.0 + rect_point2.x,
                                                    (rect_point1.y - rect_point2.y) / 2.0 + rect_point2.y,
                                                    height-0.2),
                                                    Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])),
                                    lifetime=rospy.Duration(300))

            id_plane_global +=1         
            Plane_marker.header.frame_id = "/map"
            Plane_markers_array.markers.append(Plane_marker)
                        
                
            PointCloud2.header.frame_id = "/map"
            point_pc2 = xyzarray_to_pc2(pc_stack, PointCloud2)

            pc_stack = np.empty((0, 3), float)



            self.pub_Lane_marker.publish(Lane_markers_array)
            self.pub_Plane_marker.publish(Plane_markers_array)
            self.lidar_pub.publish(point_pc2)
                

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('Lane_detection_node', anonymous=True)

    lane_detection = lane_detection_class()

    print("\nLane_detection_node start")

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
