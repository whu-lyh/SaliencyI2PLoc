

import os
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_data(batch_data):
    '''Normalize the batch data, use coordinates of the block centered at origin.
        Input:
            BxNxC array
        Output:
            BxNxC array
    '''
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def apply_transform(points: np.ndarray, transform: np.ndarray):
    """Apply transform to input point clouds.

    Args:
        points (np.ndarray): input point clouds, Nx3
        transform (np.ndarray): transformation matrix

    Returns:
        points: transformerd point clouds
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    return points


def apply_transform4x4(points: np.ndarray, transform: np.ndarray):
    """Apply transform to input point clouds.

    Args:
        points (np.ndarray): input point clouds, Nx4
        transform (np.ndarray): transformation matrix 4x4

    Returns:
        points: transformerd point clouds
    """
    xyz = points[:, :3]
    intensities = points[:, 3].reshape(-1, 1)
    ones = np.ones((xyz.shape[0], 1))
    homogeneous_points = np.hstack((xyz, ones))
    trans_homo_points = homogeneous_points @ transform.T
    transformed_points = np.hstack((trans_homo_points[:, :3], intensities))
    return transformed_points


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """Get random rotation matrix based on given rotation_factor

    Args:
        rotation_factor (float, optional): the random degree. Defaults to 1.0.

    Returns:
        np.ndarray: rotation matrix
    """
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    """Get random transformation matrix based on given rotation and translation magnitude

    Args:
        rotation_magnitude (float): rotation magnitude
        translation_magnitude (float): translation magnitude

    Returns:
        np.ndarray: _description_
    """
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


def load_pc_file(filename:str, pts_num:int=0, augment:bool=False):
    '''
        return Nx3 matrix from a .bin file (KITTI style point cloud file)
        data augmentation of random rotation along z-axis will decline the model FIXME
    '''
    pc = np.fromfile(os.path.join('', filename), dtype=np.float32) # roc edit 64->32
    pc = np.reshape(pc, [-1, 4])
    if pts_num != 0:
        n = np.random.choice(len(pc), pts_num, replace=False)
        pc = pc[n]
    # only xyz is used here
    pc = pc[:, :3]
    return rotate_point_cloud_z(pc) if augment else pc


def load_npy_file(filename:str, crop_dist:float=50.0, pts_num:int=0):
    '''
        return Nx3 matrix from a .npy file (vivid style point cloud file)
        Check whether to crop the far distance points
    '''
    points = np.load(filename).astype(np.float32) # shape is nx3, 32 is vital
    if crop_dist > 0.0:
        dis = np.linalg.norm(points, axis=1)
        points = points[np.where(dis < crop_dist)[0]]
    if pts_num != 0:
        n = np.random.choice(len(points), pts_num, replace=False)
        points = points[n]
    return points


def loadCloudFromBinary(filename:str):
    '''
        return Nx4 matrix from a .bin file (KITTI or KITTI360 style point cloud file)
    '''
    bin_pcd = np.fromfile(filename, dtype=np.float32)
    points = bin_pcd.reshape(-1, 4)
    return points


def load_pc_file_with_intensity(filename:str, pts_num:int=0):
    '''
        return Nx4 matrix from a .bin file (KITTI or KITTI360 style point cloud file)
    '''
    pc = np.fromfile(os.path.join('', filename), dtype=np.float32) # roc edit 64->32
    pc = np.reshape(pc, [-1, 4])
    # randomly select pts_num points here which means that 
    if pts_num != 0:
        n = np.random.choice(len(pc), pts_num, replace=False)
        pc = pc[n]
    return pc


def load_pc_file_fix_size(filename:str, pts_limit:int=None, 
                          augment:bool=False, rotation_magnitude:float=1.0, translation_magnitude:float=2.0):
    '''
        return Nx3 matrix from a .bin file (KITTI or KITTI360 style point cloud file)
        but with optional data argumentation, the final point number will no exceed pts_limit
    '''
    # load from bin file
    pc = np.fromfile(os.path.join('', filename), dtype=np.float32) # roc edit 64->32
    pc = np.reshape(pc, [-1, 4])
    points = pc[:, :3] # xyz only
    # voxel down sample, currently the point submap has already downsampled!
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud = point_cloud.voxel_down_sample(0.3)
    # points = np.array(point_cloud.points)
    # get the limited number of point clouds
    if pts_limit is not None and points.shape[0] > pts_limit:
        indices = np.random.permutation(points.shape[0])[:pts_limit]
        points = points[indices]
    # data argumentation
    if augment:
        transform = random_sample_transform(rotation_magnitude, translation_magnitude)
        points = apply_transform(points, transform)
    return points.astype(np.float32)


def load_pc_files(filenames:list, pts_num:int=4096, augment:bool=False):
    '''
        Load point cloud from a branch of files, merge all into one single file
        the final point number should be pts_num, by default=4096
        data augmentation of random rotation along z-axis will decline the model FIXME
    '''
    # initialize the out put list container
    pcs = []
    for filename in filenames:
        # point number check
        pc = load_pc_file(filename, pts_num, augment)
        if pc.shape[0] != pts_num:
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def load_npy_files(filenames:list, crop_dist:float=50.0, pts_num:int=0):
    '''
        Load point cloud from a branch of files, merge all into one single file
        return Nx3 matrix from a .npy file (vivid style point cloud file)
        Check whether to crop the far distance points
    '''
    # initialize the out put list container
    pcs = []
    for filename in filenames:
        # point number check
        pc = load_npy_file(filename, crop_dist, pts_num)
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def load_pc_files_fix_size(filenames:list, pts_limit:int=None, 
                          augment:bool=False, rotation_magnitude:float=1.0, translation_magnitude:float=2.0):
    '''
        Load point cloud from a branch of files, merge all into one single file
        but with optional data argumentation
        data augmentation of random rotation along z-axis will decline the model
    '''
    # initialize the out put list container
    pcs = []
    for filename in filenames:
        # point number check
        pc = load_pc_file_fix_size(filename, pts_limit, augment, rotation_magnitude, translation_magnitude)
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def rotate_point_cloud(batch_data):
    '''
        Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, rotated batch of point clouds
    '''
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    '''
        Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, rotated batch of point clouds
    '''
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data
