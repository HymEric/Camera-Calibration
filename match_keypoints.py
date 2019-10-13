# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 0012 14:49
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : match_keypoints.py
# @Software: PyCharm

import argparse
from sift import *
from sift_match import *

def main():
    parser = argparse.ArgumentParser(description="Match keypoints");
    parser.add_argument('in_fname_1', type=str, help='input keypoint file')
    parser.add_argument('in_fname_2', type=str, help='input keypoint file')
    parser.add_argument('out_fname', type=str, help='output matches filename')
    parser.add_argument('--lowe_threshold', nargs='?', type=float, default=0.6, help='Use David Lowe\'s ratio criterion for pruning bad matches')
    parser.add_argument('--homography_threshold', nargs='?', type=float, default=0, help='fit a homography to matches and prune by this reprojection error threshold')

    args = parser.parse_args()

    (kp1, desc1) = read_features(args.in_fname_1)
    (kp2, desc2) = read_features(args.in_fname_2)

    matches = match_features(desc1, desc2, args.lowe_threshold)

    if args.homography_threshold > 0:
        [matches, H] = filter_matches_by_homography(kp1, kp2, matches, args.homography_threshold)

    write_matches(args.out_fname, matches)

if __name__ == "__main__":
    main()

# test
# python match_keypoints.py --in_fname_1 ./chessboards_imgs/left01.jpg --in_fname_2 ./chessboards_imgs/left01.jpg --out_fname match_left01_to_left02.txt