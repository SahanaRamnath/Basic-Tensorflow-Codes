'''
---------------------------------------------
RNN-(SINGLE LAYERED)LSTM IMPLEMENTED ON MNIST
---------------------------------------------
'''

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# Data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
Xtest,ytest=mnist.test.images,mnist.test.labels

# Session
s=tf.InteractiveSession()

# Every row of the image is considered as a sequence of pixels
# So each image has 28 sequences of 28 steps


# Hyperparameters

#learning rate
alpha=0.01
#number of iterations for training
num_iter=2000
#batch size
batch_size=50

# Network Parameters

#number of sequences per image
num_input=28
#number of timesteps
timesteps=28
#output dimension/number of classes
num_classes=10
#dimension of hidden layer
h_len=150

# Random initialisation of weights
W=tf.Variable(tf.random_normal([h_len,num_classes],stddev=0.1))
b=tf.Variable(tf.random_normal([num_classes],stddev=0.1))

# Defining placeholders

#input
X=tf.placeholder(tf.float32,[None,timesteps,num_input])

#output
y=tf.placeholder(tf.float32,[None,num_classes])

# Modify the image to get sequences
# Current shape of data : [batch_size,timesteps,num_input]
# Required data : 'timetseps' number of tensors of shape [batch_size,num_input]
X_modified=tf.unstack(X,num=timesteps,axis=1)

# LSTM cell

#using a single hidden layer/LSTM in this network
lstm_cell=rnn.BasicLSTMCell(num_units=h_len,forget_bias=1.0)

#output and cell state of the LSTM
lstm_output,states=rnn.static_rnn(cell=lstm_cell,inputs=X_modified,dtype=tf.float32)

#using the last timestep's output
logits=tf.add(tf.matmul(lstm_output[-1],W),b)
#prediction
pred=tf.nn.softmax(logits)
#loss and optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer=tf.train.AdamOptimizer(alpha).minimize(loss)

# Evaluation
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initialise variables
s.run(tf.global_variables_initializer())

# Start training

for i in range(num_iter) : 

	batch_x,batch_y=mnist.train.next_batch(batch_size)

	#reshape the data to get 'batch_size' number of data points each of which has 28 sequences of 28 timesteps each
	batch_x=batch_x.reshape((batch_size,timesteps,num_input))

	s.run(optimizer,feed_dict={X:batch_x,y:batch_y})

	if i%100==0 : 
		train_accuracy=s.run(accuracy,feed_dict={X:batch_x,y:batch_y})
		print i,'\t',train_accuracy

Xtest=Xtest.reshape((Xtest.shape[0],timesteps,num_input))
test_accuracy=s.run(accuracy,feed_dict={X:Xtest,y:ytest})
print '\n\nTest Accuracy : ',test_accuracy


#alternately can use all the timestep's outputs as a list
#logits_all=[tf.add(tf.matmul(output,W),b) for output in lstm_output]







