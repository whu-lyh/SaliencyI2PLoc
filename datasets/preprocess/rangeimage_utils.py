
import numpy as np
from PIL import Image


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection, range image. From OverlapNet
    Args:
        current_vertex: raw point clouds
    Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    Formulations:
    Spherical projection
        $$
        r=\sqrt{x^2+y^2+z^2}
        $$

        $$
        \left(\begin{array}{l}
        u \\
        v
        \end{array}\right)=\left(\begin{array}{c}
        \frac{1}{2}\left[1-\arctan (y, x) \pi^{-1}\right] *w \\
        {\left[1-\left(\arcsin \left(z r^{-1}\right)+\mathrm{f}_{\mathrm{up}}\right) \mathrm{f}^{-1}\right] *h}
        \end{array}\right)
        $$
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    mask = (depth > 0) & (depth < max_range)
    current_vertex = current_vertex[mask]  # get rid of [0, 0, 0] points
    depth = depth[mask]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / (depth + 1e-8))

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity
    # tensor
    # circular convolutions? remove overlap?
    return proj_range, proj_vertex, proj_intensity, proj_idx

def createRangeImage(points, crop=True, crop_distance=False, distance_threshold=100):
    if crop_distance:
        dis = np.linalg.norm(points[:, :3], axis=1)
        points = points[np.where(dis < distance_threshold)[0]]
    image, _, _, _ = range_projection(points, fov_up=15.0, fov_down=-25.0, proj_H=64, proj_W=900) # KITTI360-submap should set another parameters
    image = Image.fromarray(np.uint8(image * 5.1), 'L')
    image = np.stack((image,) * 3, axis=-1)
    if crop:
        clipStart = image.shape[1] // 3      
        image = image[0:32, clipStart:clipStart*2, :]
    return image


def loadCloudFromBinary(file, cols=3):
    bin_pcd = np.fromfile(file, dtype=np.float32)
    points = bin_pcd.reshape(-1, 4)
    return points


def saveCloudToPLYBinary(file, pc):
    # FIXME INTENSITY IS LOST
    import open3d as o3d
    pc = pc[:, :3]
    intensity = pc[:, 3:]
    pcdt = o3d.t.geometry.PointCloud()
    pcdt.point.positions = o3d.core.Tensor(pc)
    # pcdt.point.intensities = o3d.core.Tensor(intensity.reshape(-1,1))
    o3d.t.io.write_point_cloud(file, pcdt)
    # pcd_nosam = o3d.geometry.PointCloud()
    # pcd_nosam.points = o3d.utility.Vector3dVector(pc)
    # # pcd_nosam.colors = o3d.utility.Vector3dVector(intensity)
    # o3d.io.write_point_cloud(file, pcd_nosam)


def loadLidarImage(data_path, save_path):
    lidar_points = loadCloudFromBinary(data_path)
    lidar_image = createRangeImage(lidar_points)
    return lidar_image


def convertPC2RI(data_path:str, out_path:str):
    out_file_img = out_path + "/" + Path(data_path).stem + ".png"
    lidar_points = loadCloudFromBinary(data_path)
    lidar_image = createRangeImage(lidar_points, crop=False)
    image = Image.fromarray(lidar_image)
    image.save(out_file_img)

def convertPC2PLY(data_path:str, out_path:str):
    out_file_pcd = out_path + "/" + Path(data_path).stem + ".ply"
    lidar_points = loadCloudFromBinary(data_path)
    saveCloudToPLYBinary(out_file_pcd, lidar_points)


if __name__ == "__main__":
    import argparse
    import glob
    import multiprocessing
    import os
    from pathlib import Path

    from tqdm.autonotebook import tqdm

    parser = argparse.ArgumentParser(description='Gen-Range-Images')
    parser.add_argument('--bin_path', type=str, default="/public/KITTI360", help='path to data')
    parser.add_argument('--ri_save_path', type=str, default="/public/KITTI360", help='path to data')
    parser.add_argument('--dataset', type=str, default="KITTI360Submap", choices=["KITTI360", "KITTI360Submap", "KITTI"], help='dataset')

    args = parser.parse_args()
    assert args.dataset in ("KITTI360", "KITTI360Submap", "KITTI")
    if args.dataset == "KITTI":
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21"]
    else:
        sequences = ["0000", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009","0010", "0018"]
        # sequences = ["0000"]

    # Create a multiprocessing pool with the number of desired processes
    num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    # for-loop of each sequences
    for seq in sequences:
        if args.dataset == "KITTI":
            # input path
            data_path = args.bin_path + f"/sequences/{seq}/velodyne/"
            assert os.path.exists(data_path)
            # get whole existed bin files
            files = glob.glob(data_path + '*.bin')
            # out path
            out_path = args.ri_save_path + f"/data_bev/{seq}/bev/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        elif args.dataset == "KITTI360":
            # input path
            data_path = args.bin_path + f"/data_3d_raw/2013_05_28_drive_{seq}_sync/velodyne_points/data/"
            assert os.path.exists(data_path)
            # get whole existed bin files
            files = glob.glob(data_path + '*.bin')
            # out path
            out_path = args.ri_save_path + f"/data_3d_arg_raw/2013_05_28_drive_{seq}_sync/bev/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        elif args.dataset == "KITTI360Submap":
            # input path
            data_path = args.bin_path + f"/data_3d_submap_raw/2013_05_28_drive_{seq}_sync/submaps/"
            assert os.path.exists(data_path)
            # get whole existed bin files
            files = glob.glob(data_path + '*.bin')
            # out path
            out_path = args.ri_save_path + f"/data_3d_submap_raw/2013_05_28_drive_{seq}_sync/range_image/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        else:
            raise NotImplementedError(f'Sorry, <{args.dataset}> dataset related function is not implemented!')

        # convertPC2RI(files[0], out_path)

        # print the input data path
        print(data_path)
        pool = multiprocessing.Pool(processes=num_processes)

        # Create a progress bar with the total number of files
        with tqdm(total=len(files)) as pbar:
            # Process the files in parallel using the pool
            for file_path in files:
                pool.apply_async(convertPC2RI, 
                                 args=(file_path, out_path), 
                                 callback=lambda _: pbar.update(1))
            # Close the pool and wait for the processes to finish
            pool.close()
            pool.join()

    print("Done!")