#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import struct
import ctypes
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import open3d

import tf as tf2
from tf import TransformListener

type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)

pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                 PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

DUMMY_FIELD_PREFIX = '__'

def get_odom(tf,frame, dest_frame):
    try:
        t = tf.getLatestCommonTime(dest_frame, frame)
        position, quaternion = tf.lookupTransform(dest_frame, frame, t)

        trans_mat = tf2.transformations.translation_matrix(position)
        rot_mat = tf2.transformations.quaternion_matrix(quaternion)
        # create a 4x4 matrix
        mat = np.dot(trans_mat, rot_mat)
    except(tf2.Exception, tf2.ConnectivityException, tf2.LookupException):
        #print("miss")
        return
    return mat

def save_ply(point,file_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point[:,:3])

    o3d.io.write_point_cloud(file_name, pcd)
    return


def get_transformation(odom_mat,points):
    odom_mat = odom_mat[:3, :]
    num_pts = points.transpose().shape[1]
    points = np.vstack((points.transpose(), np.ones((1, num_pts))))
    points = np.matmul(odom_mat, points)
    points = points.transpose()
    return points


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def xyzarray_to_pc2(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    header = parent_frame.header #std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

def xyziarray_to_pc2(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzi')]

    header = parent_frame.header #std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4),
        row_step=(itemsize * 4 * points.shape[0]),
        data=data
    )

def create_cloud_xyzi(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]
    return create_cloud(header, fields, points)

def create_cloud_xyz(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]
    return create_cloud(header, fields, points)

def create_cloud(header, fields, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message.

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param fields: The point cloud fields.
    @type  fields: iterable of L{sensor_msgs.msg.PointField}
    @param points: The point cloud points.
    @type  points: list of iterables, i.e. one iterable for each point, with the
                   elements of each iterable being the values of the fields for 
                   that point (in the same order as the fields parameter)
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step

    return pc2.PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=cloud_struct.size,
                       row_step=cloud_struct.size * len(points),
                       data=buff.raw)

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.

    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
    '''
    if remove_nans:
         mask_ = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'] & np.isfinite(cloud_array['intensity'])
         cloud_array = cloud_array[mask_]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
    '''
    # remove crap points
    # if remove_nans:
    #     mask_ = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'] & np.isfinite(cloud_array['intensity'])
    #     cloud_array = cloud_array[mask_]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']

    return points

def extract_points(points, voxel_size = 0.01, x_range= (-5, 5), y_range= (-2.2, 5.8), z_range= (-5, -1.2), i_range= (2, 8)):
    import pdb 
    pdb.set_trace() 
#x_range= (-5, 5), y_range= (-2.2, 5.8),
    x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    i = i*255

    mean = i.mean()
    std = i.std()

    i_range = (mean + i_range[0] * std, mean + i_range[1]* std)

    in_range = box_in_range(x,y,z,i, x_range, y_range, z_range, i_range)

    points = points[in_range]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:,:3])
    
    pcd = open3d.voxel_down_sample(pcd, voxel_size)
    pts_3d = np.asarray(pcd.points).astype(np.float32)

    return pts_3d

def box_in_range(x,y,z,i, x_range, y_range, z_range, i_range):
    """ extract filtered in-range velodyne coordinates based on x,y,z limit """
    return np.logical_and.reduce((
            x > x_range[0], x < x_range[1],
            y > y_range[0], y < y_range[1],
            z > z_range[0], z < z_range[1],
            i > i_range[0], i < i_range[1]))


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
