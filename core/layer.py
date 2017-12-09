# -*- coding:utf-8 -*-

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
        self.activator      = get_activator(activator_type)
        self.output_shape   = self.get_output_shape()
        self.activator_grds = np.zeros(self.output_shape, dtype=np.float64)

        self.bias           = np.zeros((self.filter_number), dtype=np.float64)
        self.bias_grads     = np.zeros((self.filter_number), dtype=np.float64)

        self.filters        = [
            np.random.uniform(-1e-4, 1e-4, [input_shape[0], filter_shape[0], filter_shape[1]])
            for _ in range(filter_number)
        ]
        self.filter_grads   = [
            np.zeros([input_shape[0], filter_shape[0], filter_shape[1]], dtype=np.float64)
            for _ in range(filter_number)
        ]
        
    def get_output_shape(self):
        chunnel = self.filter_number
        height  = int((self.input_shape[1] - self.filter_shape[0] + 2*self.padding_size[0]) / self.stride[0] + 1 )
        width   = int((self.input_shape[2] - self.filter_shape[1] + 2*self.padding_size[1]) / self.stride[1] + 1 )
        return [chunnel, height, width]

    def forward_pass(self, input_map):
        #输入类型检查和shape大小匹配
        assert(input_map.shape == tuple(self.input_shape))
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)
        
        #计算当前层输出结果    
        output_map = np.zeros(self.output_shape, dtype=np.float64)
        input_map  = padding_0(input_map, self.padding_size)
        for f in range(self.filter_number):
            for d in range(self.input_shape[0]):
                output_map[f, :, :] += convalution_2d(
                    input_map[d, :, :], self.filters[f][d, :, :], 
                    self.stride , self.bias[f]
                )
            element_wise(self.activator, output_map[f, :, :])

        #获得当前层激活函数的导数
        self.activator_grds = np.array(output_map, dtype=np.float64)
        element_wise(
            lambda x : 1 if x > 0 else 0, self.activator_grds
        )

        return output_map

    #向后传播，
    def backward_pass(self, input_map, next_layer_sensitive):
        #输入类型检查和shape大小匹配
        assert(input_map.shape == tuple(self.input_shape))
        assert(next_layer_sensitive.shape == tuple(self.output_shape))
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)

        # 计算当前层的detas
        detas = np.multiply(next_layer_sensitive, self.activator_grds)
        # detas = np.zeros(self.output_shape, dtype=np.float64)
        # next_layer_chunnel_number = next_layer_sensitive.shape[0]
        # print(next_layer_sensitive.shape, self.activator(5))
        # for f in range(self.filter_number):
        #     for d in range(next_layer_chunnel_number):
        #         detas[f, :, :] += np.multiply(
        #             self.activator_grds[f, :, :],
        #             next_layer_sensitive[d, :, :] 
        #         )

        # 计算当前层filters的权值梯度
        pd_height = self.input_shape[1] - self.filter_shape[0] + 1
        pd_width  = self.input_shape[2] - self.filter_shape[1] + 1
        pd_shape  = [pd_height, pd_width]
        # detas = np.ones(self.output_shape) #for debug
        detas = extend_map_with_stride1(detas, pd_shape, self.stride)
        for f in range(self.filter_number):
            self.bias_grads[f] = detas[f].sum()
            for d in range(self.input_shape[0]):
                self.filter_grads[f][d, :, :] = convalution_2d(
                    input_map[d, :, :], detas[f], [1,1], 0
                )

        #计算向后层的对应的detas要的关于当前层的值
        cur_layer_sensitive = []
        pd_shape  = [self.filter_shape[0]-1, self.filter_shape[1]-1]
        for d in range(self.input_shape[0]):
            cur_map  = np.zeros([self.input_shape[1], self.input_shape[2]], dtype=np.float64)
            for f in range(self.filter_number):
                pd_detas = padding_0(detas[f], pd_shape)
                rotate_filter = np.rot90(self.filters[f][d, :, :], 2)
                cur_map += convalution_2d(
                    pd_detas, rotate_filter, [1,1], 0
                )
            cur_layer_sensitive.append(cur_map)

        # 更新权值
        for f in range(self.filter_number):
            self.bias[f]    -= self.learning_rate * self.bias_grads[f]
            self.filters[f] -= self.learning_rate * self.filter_grads[f]

        return np.array(cur_layer_sensitive, dtype=np.float64)
        
class PoolingLayer(Layer):
    def __init__(self, input_shape, pooling_shape, stride, padding_size, pooling_type):
        self.stride        = stride
        self.input_shape   = input_shape
        self.padding_size  = padding_size
        self.pooling_shape = pooling_shape
        self.output_shape  = self.get_output_shape()
        self.index_map     = np.zeros(self.output_shape, dtype=tuple)
        
    def get_output_shape(self):
        width  = ceil((self.input_shape[1] - self.pooling_shape[0] + 2*self.padding_size[0]) / self.stride[0]) + 1
        height = ceil((self.input_shape[2] - self.pooling_shape[1] + 2*self.padding_size[1]) / self.stride[1]) + 1
        return [self.input_shape[0], height, width] 

    def forward_pass(self, input_map):
        #输入类型检查和shape大小匹配
        assert(input_map.shape == tuple(self.input_shape))
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)

        #计算当前层输出
        ph, pw     = self.pooling_shape
        input_map  = padding_0(input_map, self.padding_size)
        output_map = np.zeros(self.output_shape, dtype=np.float64)
        _, in_height, in_width  = input_map.shape
        depth, height, width    = self.output_shape
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
        #输入类型检查和shape大小匹配
        assert(input_map.shape == tuple(self.input_shape))
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)
        
        #这里只需要计算向后层需要的sensitive
        cur_layer_sensitive  = np.zeros(self.input_shape, dtype=np.float64)
        depth, height, width = next_layer_sensitive.shape
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    max_id = self.index_map[d,h,w]
                    cur_layer_sensitive[max_id] = next_layer_sensitive[d,h,w]

        return cur_layer_sensitive
        
class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape, learning_rate, activator_type):
        self.activator      = get_activator(activator_type)
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.learning_rate  = learning_rate
        self.activator_grds = None
        
        self.bias           = np.zeros((output_shape), dtype=np.float64)
        self.weights        = np.random.uniform(-1e-4, 1e-4, (output_shape, input_shape))
        self.bias_grads     = np.zeros(output_shape, dtype=np.float64)
        self.weights_grads  = np.zeros([output_shape, input_shape], dtype=np.float64)

    def forward_pass(self, input_map):
        #输入类型检查和shape大小匹配
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)
        input_map = input_map.flatten()
        assert(input_map.shape[0] == self.input_shape)

        #计算当前层输出
        output_map = []
        for i in range(self.output_shape):
            n_put = np.multiply(self.weights[i], input_map).sum() + self.bias[i]
            output_map.append(self.activator(n_put)) 
        output_map = np.array(output_map, dtype=np.float64)

        #计算当前层激活函数导数
        self.activator_grds = np.array(output_map, dtype=np.float64)
        element_wise(
            lambda x : 1 if x > 0 else 0, self.activator_grds
        )
        
        return output_map
    
    def backward_pass(self, input_map, next_layer_sensitive):
        # 类型检查和shape匹配
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)
        org_shape = input_map.shape
        input_map = input_map.flatten()
        next_layer_sensitive = next_layer_sensitive.flatten()
        assert(input_map.shape[0] == self.input_shape)
        assert(next_layer_sensitive.shape[0] == self.output_shape)

        # 计算本层的detas
        detas = np.multiply(next_layer_sensitive, self.activator_grds)
        
        # 计算本层的梯度
        for i in range(self.output_shape):
            self.bias_grads[i] = detas[i].sum()
            self.weights_grads[i, :] = detas[i] * input_map
        
        # 计算传递到向后层的偏导 sensitive
        cur_layer_sensitive = np.zeros_like(input_map, dtype=np.float64)
        for i in range(self.input_shape):
            cur_layer_sensitive[i] = (detas * self.weights[:, i]).sum()
        cur_layer_sensitive = cur_layer_sensitive.reshape(org_shape)

        #更新本层的权值
        self.bias    -= self.learning_rate * self.bias_grads
        self.weights -= self.learning_rate * self.weights_grads

        return cur_layer_sensitive
