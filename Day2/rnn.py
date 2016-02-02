import theano
from theano import tensor as T
from theano import shared
import numpy as np


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return shared(floatX(np.random.randn(*shape) * 0.01))


def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        g_clip = T.clip(g, -5, 5)    # clip to mitigate exploding gradients
        acc = shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g_clip ** 2
        grad_scaling = T.sqrt(acc_new + epsilon)
        g_clip = g_clip / grad_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g_clip))
    return updates


def step(x_w_xh, previous, w_hh):
    hidden = T.tanh(T.dot(previous, w_hh) + x_w_xh)
    output = T.nnet.softmax(T.dot(hidden, w_o) + b_o)
    return [hidden, output]


def model(X, init_state, w_xh, w_hh, b_h, w_o, b_o):
    x_w_xh = T.dot(X, w_xh) + b_h
    [hidden, output], _ = theano.scan( step, sequences=[x_w_xh], outputs_info=[init_state, None], non_sequences=[w_hh])

    return output, hidden[-1]


def generate(X, previous, w_xh, w_hh, b_h, w_o, b_o):
    x_w_xh = T.dot(X, w_xh) + b_h
    hidden = T.tanh(T.dot(previous, w_hh) + x_w_xh)
    output = T.nnet.softmax(T.dot(hidden, w_o)+b_o)
    return output, hidden


# data I/O
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
data_idx = [char_to_idx[ch] for ch in data]

# model parameters
hidden_size = 100    # size of hidden layer
seq_length = 25    # number of steps to unroll the RNN
lr = 1e-3    # learning rate

# build model
X = T.matrix()
init_state = T.matrix()
Y = T.matrix()

w_xh = init_weights((vocab_size, hidden_size))
w_hh = init_weights((hidden_size, hidden_size))
w_o = init_weights((hidden_size, vocab_size))
b_h = theano.shared(np.zeros((hidden_size,), dtype=theano.config.floatX))
b_o = theano.shared(np.zeros((vocab_size,), dtype=theano.config.floatX))

output, hidden = model(X, init_state, w_xh, w_hh, b_h, w_o, b_o)
# reshape output tensor from (seq_length, batch_size(1), class_num) to 2D tensor
output = T.reshape(output, (output.shape[0]*output.shape[1], output.shape[2]))
p_y = T.argmax(output, axis=1)
params = [w_xh, w_hh, b_h, w_o, b_o]
cost = T.sum(T.nnet.categorical_crossentropy(output, Y))
updates = RMSprop(cost, params, lr=lr)

train = theano.function(inputs=[X,init_state,Y], outputs=[cost,hidden], updates=updates, allow_input_downcast=True)

# for generating sequence
g_output, hidden = generate(X, init_state, w_xh, w_hh, b_h, w_o, b_o)
predict = theano.function(inputs=[X, init_state], outputs=[g_output, hidden], allow_input_downcast=True)


# start training
print "Training ..."

MAX_EPOCH = 100
for i in range(MAX_EPOCH):
    record = 0
    count = 0
    previous_value = np.zeros((1, hidden_size))

    for start, end in zip(range(0, data_size-1, seq_length), range(seq_length, data_size-1, seq_length)):

        # encode input/target as 1-of-k representation
        inputs = data_idx[start:end]
        targets = data_idx[start + 1:end + 1]
        x = np.zeros((seq_length, vocab_size))
        y = np.zeros((seq_length, vocab_size))
        for t in xrange(len(inputs)):
            x[t, inputs[t]] = 1
            y[t, targets[t]] = 1

        # train model
        cost, previous_value = train(x, previous_value, y)

        record += cost
        count += 1

    print "Epoch %d, Cost = %f" % (i + 1, record / count)


print "Start generating text ... "
x = np.zeros((1, vocab_size))
x[0,np.random.randint(vocab_size)] = 1
previous_value = np.zeros((1, hidden_size))
MAX_LEN = 1000
text = [idx_to_char[int(np.argmax(x))]]

for i in range(MAX_LEN):
    x, previous_value = predict(x, previous_value)
    text.append(idx_to_char[int(np.argmax(x))])

fout = open('generated_text.txt', 'w')
fout.write(''.join(text))
fout.close()
