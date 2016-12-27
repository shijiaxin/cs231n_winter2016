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
#### Update rules
可以参考http://sebastianruder.com/optimizing-gradient-descent/  

SGD公式如下:  
$v=- r * dW;$  
$W=W+v;$  

SGD+Momentum则是表明物体有一定惯性朝着原有方向前进，公式如下:  
$v=\gamma *v-r * dW;$  
$W=W+v;$  
这里 $\gamma$ 表示原有速度损失了一定比例  

RMSprop的公式如下：  
$Cache[g^2]=0.9*Cache[g^2]+0.1*g_t^2$  
$W=W-\frac{\eta}{\sqrt{Cache[g^2]+ \epsilon}} * g_t$  

Adam公式如下:  
$m_t=β_1m_{t−1}+(1−β_1)g_t$  
$v_t=β_2v_{t−1}+(1−β_2)g_t^2$  
$\hat{m}_t=\frac{m_t}{1-β_1^t}$  
$\hat{v}_t=\frac {v_t} {1-β_2^t}$  
$W=W-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}*\hat{m}_t$

# Batch Normalization
如果不考虑back forward，计算out其实不难，公式如下:  
sample_mean=x.mean(axis=0);  
sample_var=((x-sample_mean)\*\*2).mean(axis=0);  
xhat=(x-sample_mean)/(np.sqrt(sample_var)+eps);  
out=xhat*gamma+beta;  
但是为了计算back forward，需要写得更加详细些，具体见代码。  
时刻记住链式法则即可：  
$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial z} * \frac{\partial z}{\partial x}$   
$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial z} * \frac{\partial z}{\partial y}$   
z是x和y的某个函数，f是z的某个函数  
因此  
$dx=dz*\frac{\partial z}{\partial x}$  
$dy=dz*\frac{\partial z}{\partial y}$  
# Dropout
np.random.rand的参数必须是整数，而不是list，无法如下调用：  
np.random.rand(x.shape)  
必须要  
np.random.rand(*x.shape)  

# Convolution Network
#### Convolution forward
np.lib.pad的第二个参数是一系列的(before,after)pair，表示该纬度的前后需要加多少个padding  
举个栗子:  
np.lib.pad(input,((0,0),(pad,pad),(pad,pad)),'constant', constant_values=0);  
这个表示input的第一个纬度不加padding，第二第三纬度都要前后加两个padding，padding内容为0  
选取ndarray的某个子ndarray时候，应该用如下格式:  
padx[:,h1:h2,w1:w2]  
而不是如下格式:  
padx[:][h1:h2][w1:w2]  

#### Convolution backward
直接的思路就是保持原有的循环，把forward的代码换成backward  
