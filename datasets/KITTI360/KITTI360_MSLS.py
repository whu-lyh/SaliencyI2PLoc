import csv
import os

import numpy as np
from Camera import CameraFisheye, CameraPerspective
from tqdm import tqdm

if __name__=="__main__":

    kitti360Path = "/public/KITTI360"
    kitti360panoPath = "/public/KITTI360"

    cam_id = 3
    seq_all = [0, 2, 3, 4, 5, 6, 7, 9, 10]

    for seq in seq_all:
        sequence = '2013_05_28_drive_%04d_sync'%seq
        # perspective
        if cam_id == 0 or cam_id == 1:
            camera = CameraPerspective(kitti360Path, sequence, cam_id)
        # fisheye, 2-left,3-right
        elif cam_id == 2 or cam_id == 3:
            camera = CameraFisheye(kitti360Path, sequence, cam_id)
            #print("camera.fi:",camera.fi)
        else:
            raise RuntimeError('Invalid Camera ID!')

        # loop over frames
        target_path = os.path.join(kitti360panoPath, "data_2d_pano", sequence)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        query_file = os.path.join(target_path, "query.csv")
        database_file = os.path.join(target_path, "database.csv")
        print('query_file:\t{}'.format(query_file))
        print('database_file:\t{}'.format(database_file))

        np.set_printoptions(suppress=True, threshold=np.inf)
        with open(query_file,'w', newline='') as f_query, open(database_file,'w',newline='') as f_databse:
            writer_query = csv.writer(f_query)
            writer_database = csv.writer(f_databse)
            header = ['', 'key', 'east', 'north']
            writer_query.writerow(header)
            writer_database.writerow(header)
            cnt_q = cnt_db = 0
            accum_dist = 0.0
            pre_frame_id = 0
            for frame in tqdm(camera.frames):
                pose = camera.cam2world[frame]
                # to make the structure same as the AE-Sperical paper
                frame_str = '%g'%(frame)
                frame_str = frame_str.ljust(10, 'a')
                if frame > 1:
                    pre_pose = camera.cam2world[pre_frame_id]
                    accum_dist = accum_dist + ((pose[0, 3] - pre_pose[0, 3])**2 + (pose[1, 3] - pre_pose[1, 3])**2)**0.5
                    if accum_dist > 3.0:
                        record = [cnt_q, frame_str, pose[0,3], pose[1,3]]
                        writer_query.writerow(record)
                        accum_dist = 0.0
                        cnt_q = cnt_q + 1
                    else:
                        record = [cnt_db, frame_str, pose[0,3], pose[1,3]]
                        writer_database.writerow(record)
                        cnt_db = cnt_db + 1
                else:
                    record = [cnt_db, frame_str, pose[0,3], pose[1,3]]
                    writer_database.writerow(record)
                    cnt_db = cnt_db + 1

                pre_frame_id = frame           
