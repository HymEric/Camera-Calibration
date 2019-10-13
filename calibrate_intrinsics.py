# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 10:17
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : calibrate_intrinsics.py
# @Software: PyCharm

import argparse
import os.path
import numpy as np
import cv2
import sift
import sift_match
import calibration_fac as calib

def matches_to_calib_pts(matches, pts1, pts2):
    assert (len(matches) > 0)
    # print(matches)
    # print(len(matches))
    #
    # print(len(pts1))
    # print(len(pts2))

    obj_pts = np.array([pts1[x.queryIdx].pt for x in matches])
    img_pts = np.array([pts2[x.trainIdx].pt for x in matches])

    # add a zero z-coordinate
    N = len(matches)
    obj_pts = np.hstack([obj_pts, np.zeros((N, 1))])

    obj_pts = obj_pts.astype(np.float32)
    img_pts = img_pts.astype(np.float32)

    return (obj_pts, img_pts)


def to_calibration_data(match_sets, obj_keys, img_keysets, min_matches):
    match_keys = zip(match_sets, img_keysets)
    match_keys = [x for x in match_keys if len(x[0]) >= min_matches]

    calib_pts = [matches_to_calib_pts(m, obj_keys, k) \
                 for (m, k) in match_keys]

    return zip(*calib_pts)


def calibrate_intrinsic(obj_pts, img_pts, img_size):
    [error, K, distortion, rvecs, tvecs] = cv2.calibrateCamera(obj_pts, img_pts, img_size,None,None)
    print('K: {}'.format(K))
    cam = calib.Intrinsic_camera()
    cam.K = K
    cam.distortion = distortion
    cam.image_size = img_size

    return cam


def main():
    parser = argparse.ArgumentParser(description="Calibrate using keypoint matches")
    parser.add_argument('--pattern_key_fname', type=str,default='keypoints_dir/left01.txt', help='reference pattern keypoint file')
    parser.add_argument('--img_keypoints_root', type=str,default='./keypoints_dir/',
                        help='Printf-formatted string representing keypoint files from N scene frames')
    parser.add_argument('--select_match_keypoints_root', type=str, default='./select_match_keypoints_dir/',help='Printf-formatted string representing keypoint match files')
    parser.add_argument('--example_image', type=str,default='./meilan_note6_resized/note01.jpg', help='Example image to get dimensions from')
    parser.add_argument('--out_fname', type=str, default='calibration_intrinsics.txt',help='filename for intrinsic calibrated camera')
    parser.add_argument('--min_matches', default=20, type=str, help='omit frames with fewer than N matches')

    args = parser.parse_args()

    pattern_keys = sift.read_features(args.pattern_key_fname)[0]

    img_keypoints_names=os.listdir(args.img_keypoints_root)
    img_keypoints_paths=[os.path.join(args.img_keypoints_root,f) for f in img_keypoints_names]

    select_match_keypoints_names = os.listdir(args.select_match_keypoints_root)
    select_match_keypoints_paths=[os.path.join(args.select_match_keypoints_root,f) for f in select_match_keypoints_names]

    if len(select_match_keypoints_names) == 0:
        print("No matching keypoint files")
        exit(1)

    missing = next((x for x in select_match_keypoints_paths if not os.path.isfile(x)), None)
    if missing is not None:
        print("File not found: %s" % missing)
        exit(1)

    print("reading keypoint from %d frames" % len(img_keypoints_paths))
    frames_keys = [sift.read_features(x)[0] for x in img_keypoints_paths]
    print("reading matches from %d frames" % len(select_match_keypoints_paths))
    frames_matches = [sift_match.read_matches(x) for x in select_match_keypoints_paths]

    # assert len(frames_keys)==len(frames_matches),"frames_keys should equal to frames_matches!"
    # print(len(frames_keys))
    # print(len(frames_matches))

    img = cv2.imread(args.example_image)
    img_size = (img.shape[1], img.shape[0])

    [obj_pts, image_pts] = to_calibration_data(frames_matches, pattern_keys, frames_keys, args.min_matches)

    cam = calibrate_intrinsic(obj_pts, image_pts, img_size)

    calib.write_intrinsic_camera(args.out_fname, cam)


if __name__ == "__main__":
    main()

# test
# python calibrate_intrinsics.py --pattern_key_fname keypoints_dir/left01.txt --img_keypoints_root ./keypoints_dir/ --select_match_keypoints_root ./select_match_keypoints_dir/ --example_image ./meilan_note6_resized/note01.jpg