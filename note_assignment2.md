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
