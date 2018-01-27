# Linear Regression using tf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#instead of draw use pause and block as in stackoverflo

#plt.ion()
n=100
#fig,ax=plt.subplots(1,1)
xs=np.linspace(-3,3,n)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n)
plt.scatter(xs,ys)
#plt.pause(0.0001)
plt.show()
#plt.show(block=True)

#defining placeholders

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_normal([1]),name='wt')
b=tf.Variable(tf.random_normal([1]),name='bias')
ypred=tf.add(tf.multiply(X,W),b)

loss=tf.pow(Y-ypred,2)
cost=tf.reduce_sum(loss)/(n-1)

#adding regularization
cost=tf.add(cost,tf.multiply(1e-6,tf.global_norm([W])))
#gradient descent
eps=0.01
opt=tf.train.GradientDescentOptimizer(eps).minimize(cost)

epoch=1000
fig1=plt.figure()
fig0=fig1.add_subplot(111)

with tf.Session() as s : 
	s.run(tf.global_variables_initializer())

	prev_traincost=0
	for i in range(epoch) : 
		for (x,y) in zip(xs,ys) : 
			s.run(opt,feed_dict={X:x,Y:y})
		traincost=s.run(cost,feed_dict={X:xs,Y:ys})
		print traincost

		if i%20==0 : 
			#plt.show(block=True)
			fig0.plot(xs,ypred.eval(feed_dict={X:xs},session=s),'r')
			#plt.show()
			#print xs,ypred.eval(feed_dict={X:xs},session=s)

		if np.abs(prev_traincost-traincost)<1e-5 : 
			break
		prev_traincost=traincost

fig0.plot(xs,ys,'bo')
plt.show()
#plt.show(block=True)

