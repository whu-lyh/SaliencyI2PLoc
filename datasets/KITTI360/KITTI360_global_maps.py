'''
This demo code is originated from here "https://github.com/cattaneod/CMRNet/blob/master/preprocess/kitti_maps.py"
'''

import argparse
import os
import sys

import numpy as np
import open3d
import RT_Reader
import torch
from tqdm import tqdm

home_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
print(home_path)
if home_path not in sys.path:
    sys.path.append(home_path)
    
from KITTI360.LoadCalibration import loadCalibrationRigid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='03', help='sequence')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
    parser.add_argument('--start', default=0, help='Starting Frame')
    parser.add_argument('--end', default=999999, help='End Frame')
    parser.add_argument('--map', default=None, help='Skip map generation and conduct submap file')
    parser.add_argument('--remove_ground', default=False, help='Remove the ground when making global map')
    parser.add_argument('--create_submap', default=False, help='Create submaps from global map')
    parser.add_argument('--save_pcd', default=False, help='Save submaps in pcd format')
    parser.add_argument('--kitti360Path', default='', help='Folder of the KITTI360 dataset')
    parser.add_argument('--submapPath', default='', help='Folder of the Submap')

    args = parser.parse_args()
    print("Sequnce: ", args.sequence)
    map_file = args.map
    # read cam0_to_world
    poses, process_index = RT_Reader.PoseReader(args.sequence, args.kitti360Path, readpose=False)
    # process_index is the image idx listed in cam0_to_world.txt
    process_index = list(process_index)
    # read cam_to_velo
    fileCameraToLidar = os.path.join(args.kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    cam2lidar = loadCalibrationRigid(fileCameraToLidar)
    cam2lidar = torch.tensor(cam2lidar).double()
    print("cam2lidar transformation matrix:\n", cam2lidar)
    lidar2cam = cam2lidar.inverse()
    print("lidar2cam transformation matrix:\n", lidar2cam)

    base_path = os.path.join(args.submapPath, f'2013_05_28_drive_00{args.sequence}_sync')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    first_frame = int(args.start)
    last_frame = min(len(poses), int(args.end))
    if args.remove_ground:       
        failed_file_path_list = os.path.join(base_path, f"failed_frames_{args.sequence}.txt")
        f = open(failed_file_path_list, "w")

    if map_file is None:
        pcl = open3d.geometry.PointCloud()
        for i in tqdm(range(first_frame,last_frame)):
            # load the "specific" .bin point cloud that with poses available
            file = os.path.join(args.kitti360Path, f'data_3d_raw/2013_05_28_drive_00{args.sequence}_sync/velodyne_points','data', '%010d.bin' % int(process_index[i]))
            if not os.path.exists(file):
                continue
            pc_ori = np.fromfile(file, dtype=np.float32)
            pc_ori = pc_ori.reshape((-1, 4))
            # remove ground for single frame
            if args.remove_ground:
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(pc_ori[:, :3])
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
                j = 0
                while np.argmax(plane_model[:-1]) != 2:
                    j += 1
                    pcd = pcd.select_by_index(inliers, invert=True)
                    # pcd = pcd.select_by_index(inliers, invert=True)
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
                    if j == 5:
                        f.write(f'{int(process_index[i]):06d}.bin\n')
                        f.flush()
                        break
                outliers_index = set(range(pc_ori.shape[0])) - set(inliers)
                outliers_index = list(outliers_index)
                pc = pc_ori[outliers_index]
            else:
                pc = pc_ori
            valid_indices = pc[:, 0] < -3.
            valid_indices = valid_indices | (pc[:, 0] > 3.)
            valid_indices = valid_indices | (pc[:, 1] < -3.)
            valid_indices = valid_indices | (pc[:, 1] > 3.)
            pc = pc[valid_indices].copy()
            intensity = pc[:, 3].copy()
            pc[:, 3] = 1.
            # RT = poses[i].numpy()
            cam2world = poses[i]
            pose = cam2world @ lidar2cam
            # print(pose)
            # transform velodyne to world, not GPS/IMU coordinate
            pc_rot = np.matmul(pose, pc.T)
            pc_rot = pc_rot.T.clone() #.astype(np.float)

            pcl_local = open3d.geometry.PointCloud()
            pcl_local.points = open3d.utility.Vector3dVector(pc_rot[:, :3])
            pcl_local.colors = open3d.utility.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)
            downpcd = pcl_local.voxel_down_sample(voxel_size=args.voxel_size)
            pcl.points.extend(downpcd.points)
            pcl.colors.extend(downpcd.colors)

        downpcd_full = pcl.voxel_down_sample(voxel_size=args.voxel_size)
        downpcd, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)  #nb_neighbors=40, std_ratio=0.3
        # open3d.visualization.draw_geometries([downpcd])
        map_path = os.path.join(args.submapPath, f'2013_05_28_drive_00{args.sequence}_sync/', f'./map-{args.sequence}_{args.voxel_size}_{first_frame}-{last_frame}.pcd')
        open3d.io.write_point_cloud(map_path, downpcd)
        print('map saved!')
    else:
        downpcd = open3d.io.read_point_cloud(map_file)

    if args.create_submap:
        voxelized = torch.tensor(downpcd.points, dtype=torch.float64)
        voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float64)), 1)
        voxelized = voxelized.t()
        voxelized = voxelized.to(args.device)
        vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float64)[:, 0:1].t()

        # save individual submaps
        bin_file_dir = os.path.join(args.submapPath, f'2013_05_28_drive_00{args.sequence}_sync/submaps')
        if not os.path.exists(bin_file_dir):
            os.mkdir(bin_file_dir)
        if args.save_pcd:
            pcd_file_dir = os.path.join(args.submapPath, f's{args.sequence}_pcd')
            if not os.path.exists(pcd_file_dir):
                os.mkdir(pcd_file_dir)
        for i in tqdm(range(first_frame, last_frame)):
            cam2world = poses[i]
            pose = cam2world @ lidar2cam
            pose = pose.to(args.device)
            pose = pose.inverse()

            local_map = voxelized.clone()
            local_intensity = vox_intensity.clone()
            # transform global to current pose position(normalization)
            local_map = torch.mm(pose, local_map).t().cpu()
            # range set(crop is very efficient)
            indexes = local_map[:, 1] > -20.  # 25
            indexes = indexes & (local_map[:, 1] < 20.)  # 25
            indexes = indexes & (local_map[:, 0] > -20.)  # -10
            indexes = indexes & (local_map[:, 0] < 20.)  # 100
            local_map = local_map[indexes]
            local_intensity = local_intensity[:, indexes]
            local_map = local_map[:, :3]
            local_intensity = np.vstack((local_intensity, local_intensity, local_intensity)).T
            # submap save in pcd
            if args.save_pcd:
                pcd_nosam = open3d.geometry.PointCloud()
                # note that the open3d.utility.Vector3dVector only accept the numpy argument.
                pcd_nosam.points = open3d.utility.Vector3dVector(local_map.to("cpu"))
                pcd_nosam.colors = open3d.utility.Vector3dVector(local_intensity)
                open3d.io.write_point_cloud(pcd_file_dir + f'/{process_index[i]:010d}.pcd', pcd_nosam)
            # save in bin
            downpcd_points = np.array(local_map.cpu(), dtype=np.float32)
            downpcd_color = local_intensity.astype(np.float32)

            np_x = downpcd_points[:, 0]
            np_y = downpcd_points[:, 1]
            np_z = downpcd_points[:, 2]
            np_i = downpcd_color[:, 0]

            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
            bin_file_path = os.path.join(bin_file_dir, f'{process_index[i]:010d}.bin')
            print("submap_dir:", bin_file_path)
            points_32.tofile(bin_file_path)
