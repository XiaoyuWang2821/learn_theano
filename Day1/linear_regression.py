import theano
from theano import tensor as T
import numpy as np

# generate data
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# build model
X = T.scalar()
Y = T.scalar()

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))    # shared variable for model weights
y = T.dot(X, w)

cost = T.mean(T.sqr(y - Y))    # loss function
gradient = T.grad(cost=cost, wrt=w)    # calculate the gradient, don't need to worry about it by now
updates = [[w, w - gradient * 0.01]]    # gradient descent with learning rate = 0.01

# compile it
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    # SGD
    record = 0
    count = 0
    for x, y in zip(trX, trY):
        cost = train(x, y)
        record += cost
        count += 1
    
    print "Average loss for iteration %d: %f" % (i, record/count)
        
print w.get_value() #something around 2

