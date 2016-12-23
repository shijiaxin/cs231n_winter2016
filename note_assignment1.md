## KNN
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
