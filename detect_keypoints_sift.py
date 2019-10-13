# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 0012 9:45
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : detect_keypoints_sift.py
# @Software: PyCharm

import argparse

from sift import *



def main():
    parser = argparse.ArgumentParser(description="Extract Sift/surf keypoints and features")
    parser.add_argument('--in_fname', type=str,default='./chessboards_imgs/left01.jpg', help='input image file')
    parser.add_argument('--out_fname', type=str,default='keypoints_dir/left01.txt', help='output filename')
    parser.add_argument('--threshold', nargs='?', type=float, default=5,
                        help='Use David Lowe\'s ratio criterion for pruning bad matches')


    args = parser.parse_args()

    (kp, desc) = detect_keypoints(imagename=args.in_fname, threshold=args.threshold)

    write_features(args.out_fname, kp, desc)

if __name__ == "__main__":
    main()

# test
# python detect_keypoints_sift.py --in_fname ./chessboards_imgs/left01.jpg --out_fname keypoints_dir/left01.txt
