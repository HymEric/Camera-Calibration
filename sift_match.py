# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 0010 21:04
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : sift_match.py
# @Software: PyCharm

from sift import *
import numpy as np
import cv2


def match_template(imagename, templatename, threshold, cutoff):
    img = cv2.imread(imagename)
    template = cv2.imread(templatename)

    [kpi, di] = detect_keypoints(imagename, threshold)
    [kpt, dt] = detect_keypoints(templatename, threshold)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(np.asarray(di, np.float32), flann_params)
    idx, dist = flann.knnSearch(np.asarray(dt, np.float32), 1, params={})
    del flann

    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices=sorted(indices,key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]

    kpi_cut = []
    for i, dis in zip(idx, dist):
        if dis < cutoff:
            kpi_cut.append(kpi[i])
        else:
            break

    kpt_cut = []
    for i, dis in zip(indices, dist):
        if dis < cutoff:
            kpt_cut.append(kpt[i])
        else:
            break

    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h1 - h2) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif + h2, :w2] = template
    newimg[:h1, w2:w1 + w2] = img

    for i in range(min(len(kpi), len(kpt))):
        pt_a = (int(kpt[i, 1]), int(kpt[i, 0] + hdif))
        pt_b = (int(kpi[i, 1] + w2), int(kpi[i, 0]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))

    cv2.imwrite('matches.jpg', newimg)

def match_features(desc1, desc2, lowe_ratio=0.6):
    """
    given descriptors got by sift.detect_keypoints for two images, return good_matches
    :param desc1: descriptors of image1
    :param desc2: descriptors of image2
    :param lowe_ratio: default 0.6 if not given
    :return: good_matches of pixel index (coordinate)
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    if desc1.shape[1] != desc2.shape[1]:
        raise Exception("incompatible feature vector dimensions")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    return good_matches

def filter_matches_by_homography(key_pts1, key_pts2, matches, threshold):
    """
    when homography_threshold >0 , use this function
    :param key_pts1: keypoints of image1
    :param key_pts2: keypoints of image2
    :param matches: got by func: match_features
    :param threshold: homography_threshold
    :return:
    """
    N = len(matches)
    i = 0

    I = [x.queryIdx for x in matches]
    src_pts = key_pts1[I,:].copy()

    I = [x.trainIdx for x in matches]
    dst_pts = key_pts2[I,:].copy()

    N = len(matches)
    [H, mask] = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

    mask = np.nonzero(np.reshape(mask, (-1)))[0]
    return ([matches[i] for i in mask], H)

def read_matches(fname):
    f = open(fname, 'r')
    if f is None:
        return None
    tokens = f.readline().split()
    assert(len(tokens) == 1)
    num_matches = int(tokens[0])

    matches = []
    for i in range(0, num_matches):
        tokens = f.readline().split()
        assert(len(tokens) == 3)
        matches.append(cv2.DMatch(int(tokens[0]), int(tokens[1]), float(tokens[2])))

    return matches

def write_matches(fname, matches):
    f = open(fname, 'w')

    f.write("%d" % len(matches))

    for m in matches:
        f.write("\n%d %d %f" % (m.queryIdx, m.trainIdx, m.distance))

def appendimages(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    w1 = img1.shape[1]
    h1 = img1.shape[0]
    w2 = img2.shape[1]
    h2 = img2.shape[0]

    max_height = np.maximum(h1, h2)

    out_img = np.zeros((max_height, w1+w2, 3), np.uint8)
    out_img[0:h1, 0:w1, :] = img1
    out_img[0:h2, w1:w1+w2, :] = img2

    return out_img

def draw_matches(img1, img2, kp1, kp2, matches):
    img_out = appendimages(img1, img2)

    x_offset = img1.shape[1]

    for match in matches:
        i1 = match.queryIdx;
        i2 = match.trainIdx;
        p1 = kp1[i1].pt
        p2 = kp2[i2].pt
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]) + x_offset, int(p2[1]))

        cv2.line(img_out, p1, p2, (0,255,0))
    return img_out

if __name__=="__main__":
    # imagename="./chessboards_imgs/left12.jpg"
    # templatename="./chessboards_imgs/left13.jpg"
    # threshold=5
    # cutoff=10
    # match_template(imagename, templatename, threshold, cutoff)

    import os

    img_root = 'keypoints_dir'
    img_names = os.listdir(img_root)
    img_paths = [os.path.join(img_root, f) for f in img_names]
    lowe_threshold = 0.6
    # print(img_paths)

    (kp1, desc1) = read_features(img_paths[0])
    for j in range(len(img_paths)):
        (kp2, desc2) = read_features(img_paths[j])
        matches = match_features(desc1, desc2, lowe_threshold)
        write_matches(
            'select_match_keypoints_dir/match_{}_to_{}.txt'.format(img_names[0].split('.')[0],
                                                                   img_names[j].split('.')[0]),
            matches)
    # for i in range(len(img_names)):
    #     (kp1, desc1) = read_features(img_paths[i])
    #     if i==len(img_names)-1:
    #         (kp2, desc2) = read_features(img_paths[0])
    #         matches = match_features(desc2, desc1, lowe_threshold)
    #         write_matches(
    #             'select_match_keypoints_dir/match_{}_to_{}.txt'.format(img_names[i].split('.')[0],
    #                                                              img_names[0].split('.')[0]),
    #             matches)
    #     else:
    #         (kp2, desc2) = read_features(img_paths[i+1])
    #         matches = match_features(desc1, desc2, lowe_threshold)
    #         write_matches(
    #             'select_match_keypoints_dir/match_{}_to_{}.txt'.format(img_names[i].split('.')[0],
    #                                                              img_names[i+1].split('.')[0]),
    #             matches)
        # for j in range(i+1,len(img_names),1):
        #     (kp2, desc2) = read_features(img_paths[j])
        #     matches = match_features(desc1, desc2, lowe_threshold)
        #     # if homography_threshold > 0:
        #     #     [matches, H] = filter_matches_by_homography(kp1, kp2, matches, args.homography_threshold)
        #     # write_matches('match_keypoints_dir/match_{}_to_{}.txt'.format(img_names[i].split('.')[0],img_names[j].split('.')[0]), matches)
        #     write_matches(
        #         'select_keypoints_dir/match_{}_to_{}.txt'.format(img_names[i].split('.')[0], img_names[j].split('.')[0]),
        #         matches)
            # read_matches(out_fname)