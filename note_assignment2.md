# Fully-Connected Neural Nets
#### Affine layer
按照assignment1的矩阵公式，很容易推导出gradient  
#### ReLU layer
dx = dout*(x>0)  
#### Two-layer network
唯一需要注意的是np.random.normal可以初始化高斯分布的数据  
#### Multilayer network
Two-Layter的时候可以写para['W1'],para['W2']这样的参数  
但是layer是变量的时候不方便  
所以改为para[('W',1)]，以tuple为key
