import layer as ly
import numpy as np
from image import *
from utils import *

class Model(object):
    def __init__(self, input_shape, output_shape):
        self.layers       = []
        self.input_shape  = input_shape
        self.output_shape = output_shape

    def addLayer(self, new_layer):
        self.layers.append(new_layer)

    def E(self, output, real):
        sen   = np.multiply(output, output - real)
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
        error, sen = self.E(cur_map, real_label)
        # print('????? => ', np.argmax(cur_map), error, cur_map)
        cur_map = sen
        indexs = [i for i in range(len(self.layers))]
        indexs.reverse()
        for i in indexs:
            print('i => ', i, '\n', cur_map)
            cur_map = self.layers[i].backward_pass(recd[i], cur_map)
        return error

    def test(self, testx, testy, codes):
        patch = 100 #testx.shape[0]
        error_times = 0
        for i in range(patch):
            cur_map = testx[i, :].reshape(1,28,28)
            for layer in self.layers:
                cur_map = layer.forward_pass(cur_map)
                # print(cur_map)
            ans = np.argmax(cur_map)
            if ans != testy[i]:
                error_times += 1
                err, sen = self.E(cur_map, codes[testy[i]])
                # print("fuck => ", ans, err, '\n', cur_map)
                print((ans, testy[i]), err, i)
        return 1.0 * error_times / patch
            

def get_iamge():
    trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
    trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
    testfile_X = '../dataset/MNIST/t10k-images.idx3-ubyte'
    testfile_y = '../dataset/MNIST/t10k-labels.idx1-ubyte'

    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    hot_codes = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    )

    dealer =  DataUtils(filename=trainfile_y)

    # print(train_X.shape)
    #以下内容是将图像保存到本地文件中
    #path_trainset = "../dataset/MNIST/imgs_train"
    #path_testset = "../dataset/MNIST/imgs_test"
    #if not os.path.exists(path_trainset):
    #    os.mkdir(path_trainset)
    #if not os.path.exists(path_testset):
    #    os.mkdir(path_testset)
    #DataUtils(outpath=path_trainset).outImg(train_X, train_y)
    #DataUtils(outpath=path_testset).outImg(test_X, test_y)

    return train_X, train_y, test_X, test_y, hot_codes, dealer

if __name__ == '__main__':
    model = Model([1,28,28],[10])
    a, b, f = init_test()
    s = b.sum()
    element_wise(
        lambda x : x / s, b
    )
    # print(s, b)
    model.addLayer( ly.Convalution2DLayer([1,28,28],16,[3,3],[1,1],[0,0],1,'relu') )
    model.addLayer( ly.PoolingLayer([16,26,26],[4,4],[4,4],[0,0],'relu') )
    model.addLayer( ly.FullyConnectedLayer(784, 100, 1, 'sigmoid'))
    model.addLayer( ly.FullyConnectedLayer(100, 10, 1, 'sigmoid'))

    # for i in range(10):
    #     print( [0.0 if j != i else 1.0 for j in range(10)] )
    print('------------------------ loading data ---------------------------')
    imge = DataUtils()
    tx, ty, testx, testy, codes, dealer = get_iamge()
    
    # model.test(tx, ty, codes)

    # for i in range(iter_times):
    #     model.train_one_sample_once(tx[0, :].reshape(1,28,28), codes[ty[0]])
    # er = model.test(tx, ty)
    # print('final : ', er)
    # for i in range(10):
    #     print(tx[i, :].reshape(28,28), ty[i], '\n')
    # print(ty[0])
    patch = 1
    iter_times = 1
    print('-----------------------   trainning   ---------------------------')
    for _ in range(iter_times):
        # error = model.test(testx, testy, codes)
        # print('fuck-> ', _, error)
        for i in range(patch):
            t = model.train_one_sample_once(tx[i, :].reshape(1,28,28), codes[ty[i], :])
            

        
    

