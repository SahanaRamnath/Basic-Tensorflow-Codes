'''
----------
SIMPLE MLP
----------
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# initialising weights randomly from N(0,1)
def initialise_weights(weight_shape) : 
	return tf.Variable(tf.random_normal(weight_shape,mean=0.0,stddev=1.0))



s=tf.InteractiveSession()


# parameters
# learning rate
alpha=0.01
# number of iterations
num_iter=2000
# batch_size
batch_size=128

# network parameters
# input layer
ip_len=784
# output layer
op_len=10
# 1st hidden layer
h1_len=256
# 2nd hidden layer
h2_len=256
# dropouts
p_h1=0.8
p_h2=0.8

# input
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)# mnist data is split into 55000 train, 10000 test, 5000 validation	

Xtrain,ytrain,Xtest,ytest=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

# input layer
X=tf.placeholder(dtype=tf.float32,shape=[None,ip_len])
# output layer
y=tf.placeholder(dtype=tf.float32,shape=[None,op_len])


W1=initialise_weights([ip_len,h1_len])
b1=initialise_weights([h1_len])

W2=initialise_weights([h1_len,h2_len])
b2=initialise_weights([h2_len])

W3=initialise_weights([h2_len,op_len])
b3=initialise_weights([op_len])



# calculate pre activation and activation for hidden layer 1
a1=tf.add(tf.matmul(X,W1),b1)
h1=tf.nn.relu(a1)
#h1=tf.nn.dropout(h1,p_h1)

# calculate pre activation and activation for hidden layer 2
a2=tf.add(tf.matmul(h1,W2),b2)
h2=tf.nn.relu(a2)
#h2=tf.nn.dropout(h2,p_h2)

# calculate pre activation and activation for output layer
a3=tf.add(tf.matmul(h2,W3),b3)
h3=tf.nn.softmax(a3)

# Calculate cross entropy directly and declare a gradient descent optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a3,labels=y))
optimizer=tf.train.AdamOptimizer(alpha).minimize(loss)

# evaluate on test set
predicted_labels=tf.argmax(h3,1)
correct_labels=tf.argmax(y,1)
correct_pred=tf.equal(predicted_labels,correct_labels)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Start training
s.run(tf.global_variables_initializer())

print 'Training with batches : \n\n'
for i in range(num_iter) : 
	batch_x,batch_y=mnist.train.next_batch(batch_size)
	s.run(optimizer,feed_dict={X : batch_x,y : batch_y})
	loss_train,accuracy_train=s.run([loss,accuracy],feed_dict={X : batch_x,y : batch_y})
	if i%20==0 : 
		print i,'     ',loss_train,'      ',accuracy_train


print '\n\nTest Accuracy : \n\n'
accuracy_test=s.run(accuracy,feed_dict={X : Xtest,y : ytest})
print accuracy_test
