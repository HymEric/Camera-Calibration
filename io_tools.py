# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 10:55
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : io_tools.py
# @Software: PyCharm

import numpy

def mat_to_str(mat):
    assert(numpy.result_type(mat) == numpy.float64)
    data = '\n'.join(' '.join(str(cell) for cell in row) for row in mat)
    return "6 %d %d\n%s" % (mat.shape[0], mat.shape[1], data)