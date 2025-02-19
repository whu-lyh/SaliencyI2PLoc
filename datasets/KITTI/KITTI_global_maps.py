# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import argparse
import os

import numpy as np
import open3d
import torch
from tqdm import tqdm

import pykitti


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat

def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat

def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
    parser.add_argument('--start', default=0, help='Starting Frame')
    parser.add_argument('--end', default=100000, help='End Frame')
    parser.add_argument('--map', default=None, help='Use map file')
    parser.add_argument('--kitti_folder', default='/public2/KITTI/Semantic_KITTI', help='Folder of the KITTI dataset')

    args = parser.parse_args()
    
    sequences = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for sequence in sequences:
        print("Sequnce: ", sequence)
        velodyne_folder = os.path.join(args.kitti_folder, 'sequences', sequence, 'velodyne')
        print("velodyne_folder: ", velodyne_folder)
        pose_path = os.path.join(args.kitti_folder, 'sequences', f'{sequence}', 'poses.txt')
        print("pose_path: ", pose_path)
        poses = np.genfromtxt(pose_path)

        map_file = args.map
        first_frame = int(args.start)
        last_frame = min(len(poses), int(args.end))
        # kitti = pykitti.odometry(args.kitti_folder, sequence)

        if map_file is None:
            pc_map = []
            pcl = open3d.geometry.PointCloud()
            for i in tqdm(range(first_frame, last_frame)):
                # pc = kitti.get_velo(i)
                pc = np.fromfile(os.path.join(velodyne_folder, f'{i:06d}.bin'), dtype=np.float32).reshape((-1, 4))
                valid_indices = pc[:, 0] < -3.
                valid_indices = valid_indices | (pc[:, 0] > 3.)
                valid_indices = valid_indices | (pc[:, 1] < -3.)
                valid_indices = valid_indices | (pc[:, 1] > 3.)
                pc = pc[valid_indices].copy()
                intensity = pc[:, 3].copy()
                pc[:, 3] = 1.
                RT = poses[i].reshape((3, 4))
                pc_rot = np.matmul(RT, pc.T)
                pc_rot = pc_rot.astype(np.float32).T.copy()

                pcl_local = open3d.geometry.PointCloud()
                pcl_local.points = open3d.utility.Vector3dVector(pc_rot[:, :3])
                pcl_local.colors = open3d.utility.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)
                downpcd = pcl_local.voxel_down_sample(voxel_size=args.voxel_size)

                pcl.points.extend(downpcd.points)
                pcl.colors.extend(downpcd.colors)

            downpcd_full = pcl.voxel_down_sample(voxel_size=args.voxel_size)
            # downpcd, ind = downpcd_full.statistical_outlier_removal(nb_neighbors=40, std_ratio=0.3)
            #open3d.draw_geometries(downpcd)
            open3d.io.write_point_cloud(os.path.join(args.kitti_folder, 'sequences', sequence, f'map-{sequence}_{args.voxel_size}_{first_frame}-{last_frame}.pcd'), downpcd)
        else:
            downpcd = open3d.io.read_point_cloud(map_file)

        # voxelized = torch.tensor(downpcd.points, dtype=torch.float)
        # voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
        # voxelized = voxelized.t()
        # voxelized = voxelized.to(args.device)
        # vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float)[:, 0:1].t()
        # # velo2cam2 = torch.from_numpy(kitti.calib.T_cam2_velo).float().to(args.device)

        # # SAVE SINGLE PCs
        # if not os.path.exists(os.path.join(args.kitti_folder, 'sequences', sequence, f'local_maps_{args.voxel_size}')):
        #     os.mkdir(os.path.join(args.kitti_folder, 'sequences', sequence, f'local_maps_{args.voxel_size}'))
        # for i in tqdm(range(first_frame, last_frame)):
        #     pose = poses[i].reshape((3, 4))
        #     pose = torch.from_numpy(pose).to(args.device)
        #     pose = pose.inverse()

        #     local_map = voxelized.clone()
        #     local_intensity = vox_intensity.clone()
        #     local_map = torch.mm(pose, local_map).t()
        #     indexes = local_map[:, 1] > -20.
        #     indexes = indexes & (local_map[:, 1] < 20.)  # 25
        #     indexes = indexes & (local_map[:, 0] > -20.)  # -10
        #     indexes = indexes & (local_map[:, 0] < 20.)  # 100
        #     local_map = local_map[indexes]
        #     local_intensity = local_intensity[:, indexes]

        #     # local_map = torch.mm(velo2cam2, local_map.t())
        #     local_map = local_map[[2, 0, 1, 3], :]

        #     file = os.path.join(args.kitti_folder, 'sequences', sequence, f'local_maps_{args.voxel_size}')
        #     bin_file_path = os.path.join(file, f'{i:06d}.bin')
        #     print("submap_dir:", bin_file_path)
        #     local_map.tofile(bin_file_path)