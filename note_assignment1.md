<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# KNN
#### compute_distances_two_loops
略  
#### predict_labels
np.argsort返回数组从小到大的索引值  
再用resize取前k个  
np.bincount可以求每个数字出现的次数  
np.argmax可以求数组最大值的索引值  
#### compute_distances_one_loop
一次求出一个test和所有train的L2距离，并赋值  
#### compute_distances_no_loops
假设test=(M,D),train=(N,D)  
test的平方，再dot乘train_ones.T,得到的是(M,N)的矩阵，每一行值都相等，为test[i]的平方和  
test_ones与train.T的平方进行dot乘，得到的也是(M,N)的矩阵，每一列都相等，为train[j]的平方和  
test与train.T进行dot乘，[i,j]位置的值为test[i]与train[j]每个元素乘积的和。  
最后的distance则是两个平方和矩阵相加减去第三个矩阵乘以2  
#### Cross-validation
略  

# softmax

#### softmax_loss_vectorized (Naive略去)
$scores=X.dot(W);$  
$exp\_scores=np.exp(scores);$    
不过实际代码用的是如下这行，计算probs时候结果不变，精度更高。  
$exp\_scores=np.exp(scores-scores.max(axis=1,keepdims=True));$  
$probs=exp\_scores/exp\_scores.sum(axis=1,keepdims=True);$  
这里probs[i][j]表示的是对于输入X[i]，得到类别j的概率  
计算loss直接按照如下函数计算即可  
$$J(W)=-1/N*\sum_{i=1}^{N} log(probs[i][y_i])$$  
难点是计算dW，把probs用score代替，变形可得  
$$J(W)=-1/N*\sum_{i=1}^{N} log(\frac{exp\_scores[i][y_i]}{sum(exp\_scores[i][:])})$$  
$$J(W)=-1/N*\sum_{i=1}^{N} [log(exp\_scores[i][y_i])-log(sum(exp\_scores[i][:]))]$$  
$$J(W)=-1/N*\sum_{i=1}^{N} [scores[i][y_i]-log(\sum_k(e^{scores[i][k]}))]$$  
我们最终的目的是求dW，可以先求dscores  
可以看到 [ ] 内的后半部分是对称的，前半部分则只涉及到影响到了 $[i][y_i]$ 的那一项  
只看 [ ]内 ，前半部分导数为1，而且只影响 $[i][y_i]$ 的那一项，后半部分求导为  
$-\frac{e^{scores[i][j]}}{\sum_k(e^{scores[i][k]})}=-probs[i][j]$  
因此转化为实际代码  
dscores=probs;  
dscores[np.arange(N),y]-=1;  
dscores/=N;  
矩阵求导有如下公式:  
**如果有Z=X.dot(Y)，那么dX=dZ.dot(Y.T), dY=X.T.dot(dZ)**  
由于Middle=X.dot(W);  
可以得到dW=X.T.dot(dMiddle);  
最后记得加regression:  
loss += 0.5 * reg * np.sum(W * W);  
dW += reg * W;  
最后实现LinearClassifier里的train和predict即可。  

# SVM
#### svm_loss_naive
可以看到，只有margin > 0的时候，loss增加，因为scores=X[i].dot(W),  
因此对scores[j]有贡献的只有W[:,j]这一列，因此可以得到dW[:,j]+=X[i]  
类似的，dW[:,y[i]]-=X[i]  

#### svm_loss_vectorized
首先列出loss的公式:  
$score=X.dot(W)$ 是分数矩阵  
$L=1/N*\sum_{i=1}^{N} \sum_{j\not=y_i} max(0,score[i][j]-score[i][y_i]+1)$  
首先先计算dscore，最后再算dW  
可以先计算出$margin[i][j]=score[i][j]-score[i][y_i]+1$  
如果margin[i][j]<=0的话，dscore[i][j]为0，否则为1  
dscore[i][$y_i$]则是负数，看它被计算过几次  
最后，类似的，求出dscore后再求dW  

# TwoLayerNet
dscore的求法和softmax完全一样，求完后，套用之前描述的矩阵求导公式即可  
需要注意的是，对于scores = H1.dot(W2)+b2  
scores=(N,C), b2=(C,)  
因此求db2的时候，每个值等于dscores对应一列的和  

# Features
略  
