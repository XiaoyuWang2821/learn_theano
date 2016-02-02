from matplotlib import pyplot as plt
from sklearn import datasets
import theano
from theano import tensor as T
from theano import shared
import numpy as np
np.random.seed(1111)

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

# model
X = T.matrix()
Y = T.ivector()

w_h = shared(floatX(np.random.randn(2, 10) * 0.01))    # input-hidden layer weight
w_o = shared(floatX(np.random.randn(10, 3) * 0.01))    # hidden-output layer weight
b = shared(np.zeros((10,), dtype=theano.config.floatX))

hidden = T.nnet.sigmoid(T.dot(X, w_h) + b)
output = T.nnet.softmax(T.dot(hidden, w_o))
p_y = T.argmax(output, axis=1)

# optimization
cost = T.mean(T.nnet.categorical_crossentropy(output, Y))
params = [w_h, w_o]
gradients = T.grad(cost=cost, wrt=params)
updates = []
for p, g in zip(params, gradients):
	updates.append([p, p-g*0.05])     # fix learning rate 0.05

train = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_y, allow_input_downcast=True)

# generate data using sklearn
x, y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0,n_classes=3,n_clusters_per_class=1, class_sep=1)

# training
training_steps = 500
data_size = x.shape[0]
batch_size = 20
for i in range(training_steps):
    for start, end in zip(range(0,data_size,batch_size), range(batch_size,data_size,batch_size)):
        cost_val = train(x[start:end], y[start:end])

    if i%25 == 0:
        print "Iter %d: %f" % (i, cost_val.mean())

# visualization (not important)
x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
plt.scatter(x[:,0],x[:,1], marker='o',c=y) 
plt.show()
