'''
------------------------
CNN USING RAW TENSORFLOW
------------------------
'''

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Session
s=tf.InteractiveSession()

# Reading MNIST image data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
Xtest,ytest=mnist.test.images,mnist.test.labels

# Weight and bias initialisation
def initialise_weights(weight_shape) : 
	return tf.Variable(tf.truncated_normal(weight_shape,stddev=0.1))
def initialise_bias(bias_shape) : 
	return tf.Variable(tf.truncated_normal(bias_shape,stddev=0.1))

# Functions for convolution and 2x2 max pooling
def conv2d(x,W) : 
	# stride=1 for the pixels
	# zero padding ensured by padding='SAME'
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def maxpool(x) : 
	# taking maximum over a 2x2 patch 
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# Hyperparameters

#learning rate
alpha=0.001
#number of iterations for training
num_iter=20000
#batch size
batch_size=50


# Network Parameters

#input layer size : image size 28 by 28
ip_len=784
#output layer size : 10 classes
op_len=10
#dropout probability : probability to keep any unit
dropout=0.8
#first convolutional layer : number of filters/depth of layer
num_filters_1=32
#first convolutional layer : patch size
patch_size_1=5
#second convolutional layer : number of filters/depth of layer
num_filters_2=64
#second convolutional layer : patch size
patch_size_2=5
#channel depth for the image data
channel_depth_input=1
#size after convolution and maxpooling
cnn_final_size=7*7
#first/only hidden layer of FC network
h_len=1024



# Random initialiation of weights
w_conv1=initialise_weights([patch_size_1,patch_size_1,channel_depth_input,num_filters_1])
b_conv1=initialise_bias([num_filters_1])
w_conv2=initialise_weights([patch_size_2,patch_size_2,num_filters_1,num_filters_2])
b_conv2=initialise_bias([num_filters_2])
w_fc1=initialise_weights([cnn_final_size*num_filters_2,h_len])
b_fc1=initialise_bias([h_len])
w_fc2=initialise_weights([h_len,op_len])
b_fc2=initialise_bias([op_len])


# Defining placeholders

#input layer
X=tf.placeholder(tf.float32,[None,ip_len])

#output layer
y=tf.placeholder(tf.float32,[None,op_len])

#dropout probability
probab_of_keeping=tf.placeholder(tf.float32)

#first convolutional layer
X_modified=tf.reshape(X,shape=[-1,28,28,1]) # modifying to [batch size,height,width,num_colour_channels]
h_conv1=tf.nn.relu(tf.add(conv2d(X_modified,w_conv1),b_conv1))
h_pool1=maxpool(h_conv1)

#second convolutional layer
h_conv2=tf.nn.relu(tf.add(conv2d(h_pool1,w_conv2),b_conv2))
h_pool2=maxpool(h_conv2)

#first fully connected layer
ip_fc1=tf.reshape(h_pool2,[-1,cnn_final_size*num_filters_2])
fc1=tf.nn.relu(tf.add(tf.matmul(ip_fc1,w_fc1),b_fc1))
fc1=tf.nn.dropout(fc1,probab_of_keeping) # applying dropout

#output layer of the CNN
logits=tf.add(tf.matmul(fc1,w_fc2),b_fc2)
predicted_probab=tf.nn.softmax(logits)

#defining loss(cross entropy) and the optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer=tf.train.AdamOptimizer(alpha).minimize(loss)

#evaluating the model
pred=tf.equal(tf.argmax(predicted_probab,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(pred,tf.float32)) # try tf.metrics.accuracy


# Initialise variables
s.run(tf.global_variables_initializer())

# Start training
for i in range(num_iter) : 
	
	batch_x,batch_y=mnist.train.next_batch(batch_size)
	s.run(optimizer,feed_dict={X : batch_x, y : batch_y, probab_of_keeping : dropout})
	
	if i%100==0 : 
		loss_train,accuracy_train=s.run([loss,accuracy],feed_dict={X : batch_x, y : batch_y, probab_of_keeping : dropout})
		print i,'\t',loss_train,'\t',accuracy_train

accuracy_test=s.run(accuracy,feed_dict={X : Xtest,y : ytest, probab_of_keeping : 1.0})
print '\n\nTest Accuracy : ',accuracy_test











