# Polynomial Regression using tf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n=100
xs=np.linspace(-3,3,n)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n)
plt.scatter(xs,ys)
plt.show()

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

ypred=tf.Variable(tf.random_normal([1]),name='pred')
power=4

for i in range(1,power+1) : 
	W=tf.Variable(tf.random_normal([1]),name='weight_%d'%i)
	ypred=tf.add(ypred,tf.multiply(W,tf.pow(X,i)))

loss=tf.pow(Y-ypred,2)
cost=tf.reduce_sum(loss)/(n-1)
#adding regularization
lamb=1e-6
cost=tf.add(cost,tf.multiply(lamb,tf.global_norm([W])))

#gd
eps=0.001 #learning rate
opt=tf.train.GradientDescentOptimizer(eps).minimize(cost)

epoch=1000
fig=plt.figure()
fig0=fig.add_subplot(111)

with tf.Session() as s : 
	s.run(tf.global_variables_initializer())
	prev_traincost=0
	for i in range(epoch) : 
		for (x,y) in zip(xs,ys) : 
			s.run(opt,feed_dict={X:x,Y:y})
		traincost=s.run(cost,feed_dict={X:xs,Y:ys})
		print traincost

		if i%50==0 : 
			fig0.plot(xs,s.run(ypred,feed_dict={X:xs}),'r')

		if i==epoch-1  : 
			fig0.plot(xs,s.run(ypred,feed_dict={X:xs}),'k')
		if np.abs(prev_traincost-traincost)<1e-5 : 
			
			print '1'
			break
		prev_traincost=traincost

fig0.plot(xs,ys,'bo')
plt.ylim(-3,3)
plt.xlim(-4,4)
plt.show()
