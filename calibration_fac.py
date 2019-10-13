# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 10:51
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : calibration_fac.py
# @Software: PyCharm

import numpy as np
import io_tools as io_

class Intrinsic_camera:
    """
    Intrinsically calibrated camera
    """
    def __init__(self):
        self.K = np.eye(3)
        self.distortion = np.zeros((1,5))
        self.image_size = (0,0)

def write_intrinsic_camera(fname, cam):
    f = open(fname, 'w')
    f.write("K:\n")
    f.write(io_.mat_to_str(cam.K))
    f.write("\ndistortion:\n")
    f.write(io_.mat_to_str(cam.distortion))
    f.write("\nimage_size:\n")
    f.write(" ".join(map(str, cam.image_size)))
    f.write("\n")
    f.close()