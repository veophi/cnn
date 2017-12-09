import numpy as np
from math import *

def element_wise(func, array):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = func(i)

def padding_0(input_map, padding_size):
    if sum(padding_size) == 0:
        return input_map
    if len(input_map.shape) == 2:
        height, width       = input_map.shape
        pd_height, pd_width = padding_size
        output_map    = np.zeros([
            height + 2 * pd_height,
            width  + 2 * pd_width
            ])
        output_map[
            pd_height: height + pd_height,
            pd_width : width  + pd_width
            ] = input_map
        return output_map
    elif len(input_map.shape) == 3:
        _, height, width       = input_map.shape
        pd_height, pd_width = padding_size
        output_map    = np.zeros([
            input_map.shape[0],
            height + 2 * pd_height,
            width  + 2 * pd_width
            ])
        output_map[
            :,
            pd_height: height + pd_height,
            pd_width : width  + pd_width
            ] = input_map
        return output_map

def convalution_2d(input_map, kernel, stride = [1,1], bias = 0):
    in_width   = input_map.shape[1]
    in_height  = input_map.shape[0]
    out_width  = ceil((in_width  - kernel.shape[1] + 1) / stride[1])
    out_height = ceil((in_height - kernel.shape[0] + 1) / stride[0])
    output_map = np.zeros([out_height, out_width])
    for i in range(out_height):
        for j in range (out_width):
            width_l  = stride[1] * j
            width_r  = width_l   + kernel.shape[1]
            height_l = stride[0] * i
            height_r = height_l  + kernel.shape[0]
            output_map[i, j] = (
                input_map[ height_l: height_r,
                           width_l : width_r ] * kernel 
            ).sum() + bias
    return output_map

def extend_map_with_stride1(org_map3d, tar_shape, stride):
    if sum(stride) == 0:
        return org_map2d
    mul_h, mul_w   = stride
    height, width  = tar_shape
    depth    = org_map3d.shape[0]
    pd_map3d = np.zeros([depth, height, width], dtype=float)
    for h in range(org_map3d.shape[1]):
        for w in range(org_map3d.shape[2]):
            pd_map3d[:, mul_h*h, mul_w*w] = org_map3d[:, h, w]
    return pd_map3d

def get_maxvalue_index(org_map2d):
    max_id = (0,0)
    for h in range(org_map2d.shape[0]):
        for w in range(org_map2d.shape[1]):
            if org_map2d[max_id] < org_map2d[h, w]:
                max_id = (h, w)
    return max_id



def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1], 
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    # b = np.array(
    #     [[[0,1,1],
    #       [2,2,2],
    #       [1,0,0]],
    #      [[1,0,2],
    #       [0,0,0],
    #       [1,2,1]]])
    b = np.array(
        [
            [
                [0,1],
                [2,2]
            ],
            [
                [1,0],
                [1,2]
            ] 
        ]
    )

    f = [
        np.array(
            [[[-1,1,0],
            [0,1,0],
            [0,1,1]],
            [[-1,-1,0],
            [0,0,0],
            [0,-1,0]],
            [[0,0,-1],
            [0,1,0],
            [1,-1,-1]]], dtype=np.float64
        ),
        np.array(
            [[[1,1,-1],
            [-1,-1,1],
            [0,-1,1]],
            [[0,1,0],
            [-1,0,-1],
            [-1,1,0]],
            [[-1,0,0],
            [-1,0,1],
            [-1,0,0]]], dtype=np.float64
            )
    ]
    return a, b, f
 

if __name__ == '__main__':
    test_conv();