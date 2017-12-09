from utils import *
import numpy as np

class Layer(object):

    def __init__(self):
        pass

    def forward_pass(self):
        pass

    def backward_pass(self, input_map, next_layer_sensitive):
        pass

class Convalution2DLayer(Layer):
    
    def __init__(self, input_shape, filter_number, filter_shape, stride, padding_size, learning_rate, activator_type):
        self.stride         = stride
        self.input_shape    = input_shape
        self.filter_shape   = filter_shape
        self.padding_size   = padding_size
        self.filter_number  = filter_number
        self.learning_rate  = learning_rate
        self.activator      = self.get_activator(activator_type)
        self.output_shape   = self.get_output_shape()
        self.activator_grds = np.zeros(self.output_shape, dtype=float)

        self.bias         = [
            np.zeros([input_shape[0]], float)
            for _ in range(filter_number)
        ]
        self.filters      = [
            np.random.uniform(-1e-4, 1e-4, [input_shape[0], filter_shape[0], filter_shape[1]])
            for _ in range(filter_number)
        ]
        self.filter_grads = [
            np.zeros([input_shape[0], filter_shape[0], filter_shape[1]], dtype=float)
            for _ in range(filter_number)
        ]

    
    def get_activator(self, activator_type = str):
        if activator_type.lower() == 'relu':
            return lambda x : x if x > 0 else 0
        else:
            return lambda x : x
        
    def get_output_shape(self):
        chunnel = self.filter_number
        height  = ceil((self.input_shape[1] - self.filter_shape[0] + 1 + self.padding_size[0]) / self.stride[0])
        width   = ceil((self.input_shape[2] - self.filter_shape[1] + 1 + self.padding_size[1]) / self.stride[1])
        return [chunnel, height, width]

    def forward_pass(self, input_map):
        output_map = np.zeros(
            self.get_output_shape(), float
        )
        if type(input_map) == 'list':
            input_map = np.array(input_map)
        input_map = padding_0(input_map, self.padding_size)

        for i in range(self.filter_number):
            for j in range(self.input_shape[0]):
                output_map[i, :, :] += convalution_2d(
                    input_map[j, :, :], self.filters[i][j, :, :], 
                    self.stride , self.bias[i][j]
                )
            element_wise(self.activator, output_map[i, :, :])

        self.activator_grds = np.array(output_map)
        element_wise(
            lambda x : 1 if x > 0 else 0, self.activator_grds
        )
        return output_map

    #向后传播，
    def backward_pass(self, input_map, next_layer_sensitive):

        # 计算当前层的detas
        detas = np.zeros(self.output_shape, dtype=float)
        next_layer_chunnel_number = next_layer_sensitive.shape[0]
        # print(next_layer_sensitive.shape, self.activator(5))
        for f in range(self.filter_number):
            for d in range(next_layer_chunnel_number):
                detas[f, :, :] += np.multiply(
                    self.activator_grds[f, :, :],
                    next_layer_sensitive[d, :, :] 
                )

        # 计算当前层filters的权值梯度
        pd_height = self.input_shape[1] - self.filter_shape[0] + 1
        pd_width  = self.input_shape[2] - self.filter_shape[1] + 1
        pd_shape  = [pd_height, pd_width]
        # detas = np.ones(self.output_shape)
        detas = extend_map_with_stride1(detas, pd_shape, self.stride)
        for f in range(self.filter_number):
            for d in range(self.input_shape[0]):
                self.filter_grads[f][d, :, :] = convalution_2d(
                    input_map[d, :, :], detas[f],
                    [1,1] , self.bias[f][d]
                )
        # print( np.array(self.filter_grads) )

        #计算向后层的对应的detas要的关于当前层的值
        cur_layer_sensitive = []
        pd_shape  = [self.filter_shape[0]-1, self.filter_shape[1]-1]
        for d in range(self.input_shape[0]):
            cur_map  = np.zeros([self.input_shape[1], self.input_shape[2]], dtype=float)
            for f in range(self.filter_number):
                pd_detas = padding_0(detas[f], pd_shape)
                rotate_filter = np.rot90(self.filters[f][d, :, :], 2)
                cur_map += convalution_2d(
                    pd_detas, rotate_filter,
                    [1,1], self.bias[f][d]
                )
            cur_layer_sensitive.append(cur_map)
        return np.array(cur_layer_sensitive)
        
class PoolingLayer(Layer):
    def __init__(self, input_shape, pooling_shape, stride, padding_size, pooling_type):
        self.stride        = stride
        self.index_map     = None
        self.input_shape   = input_shape
        self.padding_size  = padding_size
        self.pooling_shape = pooling_shape
        self.output_shape  = self.get_output_shape()
        
    def forward_pass(self, input_map):
        assert(input_map.shape == tuple(self.input_shape))

        input_map = padding_0(input_map, self.padding_size)
        input_map = np.array(input_map)
        # print(input_map)
        ph, pw = self.pooling_shape
        _, in_height, in_width  = input_map.shape
        depth, height, width    = self.output_shape
        self.index_map = np.zeros(self.output_shape, dtype=tuple)
        output_map = np.zeros(self.output_shape, dtype=float)
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    hl, wl = h * ph, w * pw
                    hr = min(in_height, hl+ph)
                    wr = min(in_width,  wl+pw)
                    temp = get_maxvalue_index(
                        input_map[d, hl:hr, wl:wr]
                    )
                    max_id = (d, temp[0] + hl, temp[1] + wl)
                    output_map[d,h,w]     = input_map[max_id]
                    self.index_map[d,h,w] = max_id
        return output_map

    def backward_pass(self, input_map, next_layer_sensitive):
        cur_layer_sensitive  = np.zeros(self.input_shape, dtype=float)
        depth, height, width = next_layer_sensitive.shape
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    max_id = self.index_map[d,h,w]
                    cur_layer_sensitive[max_id] = next_layer_sensitive[d,h,w]
        return cur_layer_sensitive

    def get_output_shape(self):
        width  = ceil((self.input_shape[1] - self.pooling_shape[0] + 1 + self.padding_size[0]) / self.stride[0])
        height = ceil((self.input_shape[2] - self.pooling_shape[1] + 1 + self.padding_size[1]) / self.stride[1])
        return [self.input_shape[0], height, width]
        
class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape,):
        pass

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
            [1,-1,-1]]], dtype=float
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
            [-1,0,0]]], dtype=float
            )
    ]
    return a, b, f
 
    
if __name__ == '__main__':
    input_map, sen, f = init_test()
    # x = PoolingLayer([3,5,5],[2,2],[0,0], 'max')
    x = Convalution2DLayer([3,5,5],2, [3,3], [2,2], [0, 0], 0.0001, 'relu')
    x.filters = f
    # print(input_map.shape)
    y = x.forward_pass(input_map)
    # print(y)
    y = x.backward_pass(input_map,sen)
    print(y)