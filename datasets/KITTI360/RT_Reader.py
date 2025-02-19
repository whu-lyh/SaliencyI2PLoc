import numpy as np
import pandas as pd
import torch


def PoseReader(seq, kitti360Path, readpose):
    df = pd.read_table(kitti360Path + f'/data_poses/2013_05_28_drive_00{seq}_sync/cam0_to_world.txt', header=None, sep=' ') # poses
    if readpose:
        df.columns = ['i', 'r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2']
    else:
        df.columns = ['i', 'r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2', '0', '0', '0', '1', 'nan']
    process_index = df['i']
    poses = []
    # i first idx in each pose recoard
    for i in range(len(df)):
        T = np.array([[df['r00'][i], df['r01'][i], df['r02'][i], df['t0'][i]],
                      [df['r10'][i], df['r11'][i], df['r12'][i], df['t1'][i]],
                      [df['r20'][i], df['r21'][i], df['r22'][i], df['t2'][i]],
                      [0,0,0,1]])
        T = torch.tensor(T)
        poses.append(T)
    return poses, process_index