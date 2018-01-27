# Logistic Regression using tf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as ip

#data
mnist=ip.read_data_sets('MNIST_data/',one_hot=True)
print mnist.train.num_examples,mnist.test.num_examples,mnist.validation.num_examples
print mnist.train.images.shape,mnist.train.labels.shape
print np.min(mnist.train.images), np.max(mnist.train.images)

#to visualise one of the digits
plt.imshow(np.reshape(mnist.train.images[100,:],(28,28)),cmap='gray')
plt.show()

n_ip=mnist.train.images.shape[1]
n_op=mnist.train.labels.shape[1]
net_ip=tf.placeholder(tf.float32,[None,n_ip])

W=tf.Variable(tf.zeros([n_ip,n_op]),name='wt')
b=tf.Variable(tf.zeros([n_op]))
net_op=tf.nn.softmax(tf.matmul(net_ip,W)+b)

Y=tf.placeholder(tf.float32,[None,n_op])

#loss function
cross_entropy=-tf.reduce_sum(Y*tf.log(net_op))
pred=tf.equal(tf.argmax(net_op,1),tf.argmax(Y,1))

accuracy=tf.reduce_mean(tf.cast(pred,"float"))#taking mean gives accuracy here
eps=0.001
opt=tf.train.GradientDescentOptimizer(eps).minimize(cross_entropy)

s=tf.Session()
s.run(tf.global_variables_initializer())

batch_size=100
epochs=20
for i in range(epochs) : 
	for batch in range(mnist.train.num_examples//batch_size) : 
		batch_xs,batch_ys=mnist.train.next_batch(batch_size)
		s.run(opt,feed_dict={net_ip:batch_xs,Y:batch_ys})
	print 'Validation Accuracy : ',s.run(accuracy,feed_dict={net_ip:mnist.validation.images,Y:mnist.validation.labels})
	print 'Test Accuracy : ',s.run(accuracy,feed_dict={net_ip:mnist.test.images,Y:mnist.test.labels}),'\n'

