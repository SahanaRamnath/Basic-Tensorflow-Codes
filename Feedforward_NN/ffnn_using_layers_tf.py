'''
--------------------------
SIMPLE MLP WITH ESTIMATORS
--------------------------
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

# input
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)# mnist data is split into 55000 train, 10000 test, 5000 validation	

Xtrain,ytrain,Xtest,ytest=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels


# define the network
def nn(xdict) : 
	X=xdict['imgs']
	layer1=tf.layers.dense(X,h1_len,activation=tf.nn.relu)
	layer2=tf.layers.dense(layer1,h2_len,activation=tf.nn.relu)
	op_layer=tf.layers.dense(layer2,op_len)
	return op_layer

# defining a model_fn using following tf estimator template
def model_fn(features,labels,mode) : 

	# Building the network
	logits=nn(features)

	# Predictions
	predicted_classes=tf.argmax(logits,axis=1)
	predicted_probabs=tf.nn.softmax(logits)

	# if prediction mode 
	if mode==tf.estimator.ModeKeys.PREDICT : 
		return tf.estimator.EstimatorSpec(mode,predictions=predicted_classes)

	# loss and optimizer
	loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
	optimizer=tf.train.AdamOptimizer(alpha)
	train_op=optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
	acc_op=tf.metrics.accuracy(labels=tf.argmax(labels,axis=1),predictions=predicted_classes)

	# return the estimator specifications
	estim_specs=tf.estimator.EstimatorSpec(mode=mode,predictions=predicted_classes,loss=loss_op,train_op=train_op,eval_metric_ops={'accuracy' : acc_op})

	return estim_specs






# Build the estimator
model=tf.estimator.Estimator(model_fn)

# Define input function for training
print 'Training : \n\n'
input_fn=tf.estimator.inputs.numpy_input_fn(x={'imgs':Xtrain},y=ytrain,batch_size=batch_size,num_epochs=None,shuffle=True)
model.train(input_fn,steps=num_iter)

# Evaluate the model
print 'Test : \n\n'
output_fn=tf.estimator.inputs.numpy_input_fn(x={'imgs':Xtest},y=ytest,batch_size=batch_size,shuffle=False)
e=model.evaluate(output_fn)

print '\nTest Accuracy : ',e['accuracy']



