# Recurrent Neural Networks

#### RNN: step forward
公式是:  
$h_t=tanh(W_hh_{t-1}+W_xx_t+b)$  
$y_t=W_yh_t$  
需要注意的是，tanh平时用得不多，并不是arctan  
$sinh(x)=\frac{e^x - e^{-x}}{2}$  
$cosh(x)=\frac{e^x + e^{-x}}{2}$  
$tanh(x)=\frac{sinh(x)}{cosh(x)}$  
$tanh'(x)=1-tanh^2(x)$  

#### RNN: step backward
了解tanh'的公式后易得  

### RNN: forward
循环地调用rnn_step_forward  
不需要保存每次的cache，需要的时候再构造。  

#### RNN: backward
唯一需要疑惑的就是每一次调用rnn_step_backward的时候  
dnext_h的值应该是多少？  
可以这么理解，函数是f(h0)=[h1,h2,h3]  
现在有了dout=[d1,d2,d3]，该怎么求dh0?  
h3的值会影响d3，因此dh3=d3  
h2的值会影响h3和d2，即可以认为有两个函数都以h2为参数，  
因此dh2应该是这两个函数的gradient的和，即d2+dprev(h3)  
