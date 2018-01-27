# Lin Al using tensors

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Creating a tensor
x=tf.linspace(-3.0,3.0,32)
s=tf.Session()
x.eval(session=s)
s.close()

#Alternatively
s=tf.InteractiveSession()
x.eval()

#Create a gaussian with x
x=tf.linspace(-3.0,3.0,100)
sigma=1.0
m=0.0
z=(1.0/sigma*tf.sqrt(2.0*np.pi))*tf.exp(tf.negative(tf.pow(x-m,2.0)/(2.0*tf.pow(sigma,2.0))))

#graph
assert z.graph is tf.get_default_graph()
plt.plot(x.eval(),z.eval())
plt.show()

z_2d=tf.matmul(tf.reshape(z,[int(tf.shape(z).eval()),1]),tf.reshape(z,[1,int(tf.shape(z).eval())]))
plt.imshow(z_2d.eval())
plt.show()

#gabor patch
x=tf.reshape(tf.sin(tf.linspace(-3.0,3.0,100)),[100,1])
y=tf.reshape(tf.ones_like(x),[1,100])
z=tf.multiply(tf.matmul(x,y),z_2d)
plt.imshow(z.eval())
plt.show()

ops=tf.get_default_graph().get_operations()
print [op.name for op in ops]

def gabor() : 
	x=tf.linspace(-3.0,3.0,100)
	sigma=1.0
	m=0.0
	z=(1.0/sigma*tf.sqrt(2.0*np.pi))*tf.exp(tf.negative(tf.pow(x-m,2.0)/(2.0*tf.pow(sigma,2.0))))
	z_2d=tf.matmul(tf.reshape(z,[int(tf.shape(z).eval()),1]),tf.reshape(z,[1,int(tf.shape(z).eval())]))
	x=tf.reshape(tf.sin(tf.linspace(-3.0,3.0,100)),[100,1])
	y=tf.reshape(tf.ones_like(x),[1,100])
	z=tf.multiply(tf.matmul(x,y),z_2d)
	return z

#DU convolution