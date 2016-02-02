from matplotlib import pyplot as plt
from sklearn import datasets
import theano
from theano import tensor as T
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

srng = RandomStreams(12345)

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
	return shared(floatX(np.random.randn(*shape) * 0.01))

def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		acc = shared(p.get_value() * 0.)
		acc_new = rho * acc + (1-rho) * g ** 2
		grad_scaling = T.sqrt(acc_new + epsilon)
		g = g/grad_scaling
		updates.append((acc, acc_new))
		updates.append((p, p-lr*g))
	return updates

def dropout(X, p=0.):
	if p > 0:
		retain_prob = 1 - p
		X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
		X /= retain_prob
	return X

def model(X, w_h1, w_h2, w_o, p_drop):
	h1 = T.nnet.relu(T.dot(X, w_h1))

	h1_drop = dropout(h1, p_drop)
	h2 = T.nnet.relu(T.dot(h1_drop, w_h2))

	h2_drop = dropout(h2, p_drop)
	output = T.nnet.softmax(T.dot(h2_drop, w_o))

	return h1, h2_drop, output

X = T.fmatrix()
Y = T.ivector()

w_h1 = init_weights((2, 10))
w_h2 = init_weights((10, 10))
w_o = init_weights((10, 3))

h1, h2, output = model(X, w_h1, w_h2, w_o, 0.5)
t_h1, t_h2, t_output = model(X, w_h1, w_h2, w_o, 0.)
p_y = T.argmax(t_output, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(output, Y))
params = [w_h1, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.0001)

train = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs = p_y, allow_input_downcast=True)

# generate data using sklearn
x, y = datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0,n_classes=3,n_clusters_per_class=1, class_sep=1.5)

training_steps = 1000
data_size = x.shape[0]
batch_size = 20
for i in range(training_steps):
    for start, end in zip(range(0,data_size,batch_size), range(batch_size,data_size,batch_size)):
        cost_val = train(x[start:end], y[start:end])

    if i%25 == 0:
        print "Iter %d: %f" % (i, cost_val.mean())

# visualization
x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
plt.scatter(x[:,0],x[:,1], marker='o',c=y) 
plt.show()
