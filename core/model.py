import layer as ly
import numpy as np
from image import *
from utils import *

np.set_printoptions(threshold=np.inf)

class Model(object):
    def __init__(self, input_shape, output_shape):
        self.layers       = []
        self.input_shape  = input_shape
        self.output_shape = output_shape

    def addLayer(self, new_layer):
        self.layers.append(new_layer)

    def save(self, file_name):
        pass
        
    def load(self, file_name):
        pass
        

    def envaluation(self, output, real):
        sen = np.array(output, dtype=np.float64)
        sen[ np.argmax(real) ] -= 1 
        error = 0.5 * (sen * sen).sum()
        return error, sen 
    
    def train_one_sample_once(self,input_map, real_label):
        #类型检查
        if type(input_map) != 'numpy.ndarray':
            input_map = np.array(input_map, dtype=np.float64)
        if type(real_label) != 'numpy.ndarray':
            real_label = np.array(real_label, np.float64)
        assert(input_map.shape == tuple(self.input_shape))
        assert(real_label.shape == tuple(self.output_shape))

        recd = []
        cur_map  = input_map
        for layer in self.layers:
            recd.append(cur_map)
            cur_map = layer.forward_pass(cur_map)

        element_wise(exp, cur_map)
        total = cur_map.sum()
        cur_map /= total

        error, sen = self.envaluation(cur_map, real_label)
        cur_map = sen
        indexs = [i for i in range(len(self.layers))]
        indexs.reverse()
        for i in indexs:
            cur_map = self.layers[i].backward_pass(recd[i], cur_map)
        return error

    def test(self, testx, testy, codes):
        patch = 300
        error_times = 0
        for i in range(patch):
            cur_map = testx[i, :].reshape(1,28,28)
            for layer in self.layers:
                cur_map = layer.forward_pass(cur_map)

            element_wise(exp, cur_map)
            total = sum(cur_map)
            cur_map /= total

            ans = np.argmax(cur_map)
            if ans != testy[i]:
                error_times += 1
                err, sen = self.envaluation(cur_map, codes[testy[i]])
        return 1.0 - 1.0 * error_times / patch
            

if __name__ == '__main__':
    model = Model([1,28,28],[10])
    model.addLayer( ly.Convalution2DLayer([1,28,28],8,[5,5],[1,1],[0,0],0.01,'relu') )
    model.addLayer( ly.PoolingLayer([8,24,24],[2,2],[2,2],[0,0],'relu') )
    model.addLayer( ly.Convalution2DLayer([8,12,12],12,[5,5],[1,1],[0,0],0.01,'relu') )
    model.addLayer( ly.PoolingLayer([12,8,8],[2,2],[2,2],[0,0],'relu') )
    model.addLayer( ly.FullyConnectedLayer(192, 64, 0.01, 'relu'))
    model.addLayer( ly.FullyConnectedLayer(64, 10, 0.01, 'softmax'))

    print('------------------------ loading data ---------------------------')
    imge = DataUtils()
    tx, ty, testx, testy, codes, dealer = get_iamge()
    
    print('-----------------------   trainning   ---------------------------')
    iter_times = 10000
    patch = tx.shape[0]
    for _ in range(iter_times):
        temp_total = 0.0
        for i in range(patch):
            if i % 1000 == 0:
                error = model.test(testx, testy, codes)
                print('correct-> ', _, error)   
            t = model.train_one_sample_once(tx[i, :].reshape(1,28,28), codes[ty[i], :])
            temp_total += t
            if i % 100 == 0 and i != 0:
                temp_total /= 100
                print('i => ', i, ' : ', temp_total)
                temp_total = 0.0
                
        
            

        
    

