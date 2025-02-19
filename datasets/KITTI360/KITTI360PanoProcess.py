import argparse
import glob
import os
import warnings

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

'''
Usage:
    python KITTI360PanoProcess.py --raw_path /workspace/kitti360 --target_pano_path /workspace/kitti360 --single_image
    python KITTI360PanoProcess.py --raw_path /workspace/kitti360 --target_pano_path /workspace/kitti360 --single_sequence
    python KITTI360PanoProcess.py --raw_path /workspace/kitti360 --target_pano_path /workspace/kitti360
'''

def resize_crop_img_cv2(img_in, ratio, crop=False):
    '''
        resize and crop image loaded by opencv
    '''
    h, w, _ = img_in.shape
    if h == 512 and w == 1024:
        # crop the meaningless border
        if crop:
            img_in = img_in[int(h * 0.2): int(h * 0.84), 0:int(w)] # 512*1024->328*1024
            return img_in
    # crop the meaningless border
    if crop:
        img_in = img_in[int(h * 0.15): int(h * 0.9), 0:int(w)]
    # FIXME
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(img_in, ratio, (wn, hn), cv2.INTER_CUBIC)
    return img_out

def resize_crop_img_pil(img_in, ratio, crop=False):
    '''
        resize and crop image loaded by PIL
    '''
    w, h = img_in.size #  width, height
    #print(img_in.size)
    if h == 512 and w == 1024:
        # crop the meaningless border
        if crop:    
            img_in = img_in.crop((0, int(h * 0.2), int(w), int(h * 0.84))) # 512*1024->328*1024
            #print(img_in.size)
        return img_in
    # crop the meaningless border
    if crop:    
        img_in = img_in.crop((0, int(h * 0.2), int(w), int(h * 0.84)))
    # FIXME
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = img_in.resize((wn, hn), 2)
    return img_out

def resize_crop_img(img_in, ratio, crop=False):
    '''
        resize and crop image to target size
        Input: 
            If the input image size is 1440*2880, guassian blur, downsample, crop then resize
            If the input image size is already 512*1024, leave it alone but only crop the empty pixel
            Note that the second type could cause the result image to be smaller
    '''
    if type(img_in) == np.ndarray: # loaded by opencv
        img_out = resize_crop_img_cv2(img_in, ratio, True)
    else: # load by PIL
        img_out = resize_crop_img_pil(img_in, ratio, True)
    return img_out

def main():
    parser = argparse.ArgumentParser(description="panorama image preprocess")
    parser.add_argument("--dataset", default="KITTI360", type=str,
                        help="KITTI360, WUHAN, SHANGHAI")
    parser.add_argument("--raw_path", type=str, required=True, default="/data-lyh2/KITTI360",
                        help="Base path storing whole data base, for KITTI360, or others")
    parser.add_argument("--target_pano_path", type=str, required=True, default="/data-lyh2/KITTI360",
                        help="The out put preprocssed panorama images, default the same base path")
    parser.add_argument("--sequence", default="3", type=int,
                        help="0, 1 and etc, for single_sequence test")
    parser.add_argument("--single_sequence", action="store_true", 
                        help="process single sequence images for debug")
    parser.add_argument("--single_image", action="store_true", 
                        help="process single image for visualization")
    args = parser.parse_args()
    print(args)  
    # define the source image path and target image path
    kitti360panoPath = args.raw_path
    kitti360panoh5Path = args.target_pano_path
    
    if args.single_image:
        if args.dataset=="KITTI360":
            seq_all = [2]
            sequence = "2013_05_28_drive_%04d_sync"%seq_all[0]
            target_path = os.path.join(kitti360panoPath, "data_2d_pano", sequence)
            seq_key = os.path.join(target_path, "pano", "data_rgb", "%010d" % 4617 + ".png") 
            # pil return np.array style format
            img_pil = Image.open(seq_key).convert("RGB")
            img_pil.save("./%010d" % 4617 + "_raw.png")
            print("image type by PIL: ", type(img_pil))
            img = resize_crop_img(img_pil, 0.5, crop=True)
            img.save("./%010d" % 4617 + "_resize.png")
        elif args.dataset=="WUHAN":
            set = 1
        elif args.dataset=="SHANGHAI":
            set = 2
        return
    
    if args.single_sequence:
        if args.dataset=="KITTI360":
            seq_all = [args.sequence]
        elif args.dataset=="WUHAN":
            set = 1
        elif args.dataset=="SHANGHAI":
            set = 2
    else:
        if args.dataset=="KITTI360":
            seq_all = [0, 2, 3, 4, 5, 6, 7, 9, 10]
        elif args.dataset=="WUHAN":
            set = 1
        elif args.dataset=="SHANGHAI":
            set = 2
    
    # loop over whole sequences
    for seq in tqdm(seq_all, desc="all sequences".rjust(15)):
        sequence = "2013_05_28_drive_%04d_sync"%seq       
        # loop over frames
        source_path = os.path.join(kitti360panoPath, "data_2d_pano", sequence)
        if not os.path.exists(source_path):
            warnings.warn("You have chosen a nonesisted image path")
        target_path = os.path.join(kitti360panoh5Path, "data_2d_pano_crop", sequence, "pano", "data_rgb")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # glob images
        images = glob.glob(os.path.join(source_path, "pano", "data_rgb") + "/*.png")
        # process each image
        for image in tqdm(images, desc="single sequence".rjust(15)):
            image_name = os.path.basename(image)
            img = cv2.imread(image)
            img = resize_crop_img(img, 0.5, crop=True)
            image_out_path = os.path.join(target_path,image_name)
            cv2.imwrite(image_out_path,img)


if __name__=="__main__":
    print("Begin!")
    main()
    print("Done!", flush=True)
    