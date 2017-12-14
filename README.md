# cnn

### 简介

- 玩具性质的用python借助numpy包实现的简单cnn

- python版本为3.6，仅依赖numpy包

- layer.py为实现cnn的各个网络层，包含了bp过程，写的比较挫，目的仅是进一步了解cnn

- model.py抽象的并不好，把输出层给放了进去，后来懒得改了

- 神经网络还是需要GPU啊，而且建议用CPP实现，python确实是慢了些

- 在mnist数据集上在迭代3次的情况下可以达到98.67的acc，因为太慢了就没有继续迭代下去
