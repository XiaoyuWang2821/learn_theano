import numpy as np
from theano import tensor as T
from theano import function
from theano import shared
np.random.seed(123)

# data
N = 400
feats = 784
D = (np.random.randn(N,feats), np.random.randint(size=N, low=0, high=2))
training_steps = 500

# build model
x = T.matrix('x')
y = T.vector('y')
w = shared(np.random.randn(feats), name='w')
b = shared(0., name='b')

p_1 = 1 / (1 + T.exp(-T.dot(x,w)-b))    # logistic function
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)    # cross entropy loss
#xent = T.nnet.binary_crossentropy(p_1, y)
cost = xent.mean() + 0.01 * (w**2).sum()    # loss = data loss + l2 regularization
gw, gb = T.grad(cost, [w,b])

train = function(inputs=[x,y],
		outputs=[prediction, xent],
		updates=((w,w-0.1*gw),(b,b-0.1*gb)))    # fix learning rate 0.1
predict = function(inputs=[x], outputs=prediction)

for i in range(training_steps):
    pred, err = train(D[0], D[1])    

    if i%25 == 0:
        print "Iter %d: %f" % (i, err.mean())
