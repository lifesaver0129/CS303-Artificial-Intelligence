import os
import numpy as np
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Definition of functions and parameters
EPOCH = 100
loss_list = []
accuracy_list = []
lambda_value = 0.0005
learning_rate = 0.1
after_learning_rate = 0.01


# Initialize the number
def init_number(bef_size, aft_size):
    return np.random.rand(bef_size, aft_size) * 2 * np.sqrt(6) / np.sqrt(bef_size + aft_size) - np.sqrt(6) / np.sqrt(
        bef_size + aft_size)


# relu forward function
def relu_forward(layer, para_w, para_b):
    return np.maximum(0, layer.reshape(layer.shape[0], para_w.shape[0]).dot(para_w) + para_b)


# softmax forward function
def soft_forward(layer, para_w, para_b):
    return np.exp(layer.reshape(layer.shape[0], para_w.shape[0]).dot(para_w) + para_b) / \
           np.sum(np.exp(layer.reshape(layer.shape[0], para_w.shape[0]).dot(para_w) + para_b), axis=1, keepdims=True)


# layer weight
def lay_weight(layer, error):
    return np.dot(layer.T, error) / 100


# layer bias
def lay_bias(error):
    return np.mean(error, axis=0, keepdims=False)


# update gradient
def update(l_rate, x, recal_x):
    return x - recal_x * l_rate


# check accuracy
def accuracy(x, y):
    return np.sum(np.argmax(x, 1) == np.argmax(y, 1)) / x.shape[0]


# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),
                                                                     encoding='latin1')
# Data preprocessing: normalize all pixels to [0,1) by dividing 256
train_images = np.asarray([x.astype(float) / 255. for x in train_images])
test_images = np.asarray([x.astype(float) / 255. for x in test_images])
train_re_label = np.zeros((train_labels.shape[0], 10)).astype(int)
train_re_label[np.arange(len(train_labels)), train_labels.astype(int)] = 1
test_re_label = np.zeros((test_labels.shape[0], 10)).astype(int)
test_re_label[np.arange(len(test_labels)), test_labels.astype(int)] = 1

# Weight initialization: Xavier
w1 = init_number(784, 300)
w2 = init_number(300, 100)
w3 = init_number(100, 10)
b1 = np.zeros(300)
b2 = np.zeros(100)
b3 = np.zeros(10)

# Training of neural network
for e in range(1, EPOCH + 1):
    for _ in range(100):
        init_layer = train_images[_ * 100:(_ + 1) * 100]
        # Forward propagation
        f_layer = relu_forward(init_layer, w1, b1)
        s_layer = relu_forward(f_layer, w2, b2)
        prob = soft_forward(s_layer, w3, b3)
        # Backward propagation
        error_layer_3 = prob - train_re_label[_ * 100:(_ + 1) * 100]
        error_layer_2 = np.dot(error_layer_3, w3.T)
        error_layer_2[s_layer == 0] = 0
        error_layer_1 = np.dot(error_layer_2, w2.T)
        error_layer_1[f_layer == 0] = 0
        recal_w3 = lay_weight(s_layer, error_layer_3) + lambda_value * w3
        recal_w2 = lay_weight(f_layer, error_layer_2) + lambda_value * w2
        recal_w1 = lay_weight(init_layer, error_layer_1) + lambda_value * w1
        recal_b3 = lay_bias(error_layer_3)
        recal_b2 = lay_bias(error_layer_2)
        recal_b1 = lay_bias(error_layer_1)
        loss = -np.sum(np.log(prob[np.arange(prob.shape[0]), train_labels[_ * 100:(_ + 1) * 100]])) / prob.shape[0] + (
                np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3)) * 0.00025
        # Gradient update
        if e < 51:
            learning_rate_value = learning_rate
        else:
            learning_rate_value = after_learning_rate
        w1 = update(learning_rate_value, w1, recal_w1)
        w2 = update(learning_rate_value, w2, recal_w2)
        w3 = update(learning_rate_value, w3, recal_w3)
        b1 = update(learning_rate_value, b1, recal_b1)
        b2 = update(learning_rate_value, b2, recal_b2)
        b3 = update(learning_rate_value, b3, recal_b3)
    # Testing for accuracy
    f_layer = relu_forward(test_images, w1, b1)
    s_layer = relu_forward(f_layer, w2, b2)
    prob = soft_forward(s_layer, w3, b3)
    accur = accuracy(prob, test_re_label)
    print('Round', e)
    print('Loss:', loss)
    print('Accuracy:', accur)
    loss_list.append(loss)
    accuracy_list.append(accur)

# Plot
plt.figure(figsize=(12, 10))
ax1 = plt.subplot(211)
ax1.plot(np.array(range(EPOCH)), np.asarray(loss_list))
plt.title("Loss & Epoch")
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.grid()
ax2 = plt.subplot(212)
ax2.plot(np.array(range(EPOCH)), np.asarray(accuracy_list))
plt.title("Accuracy & Epoch")
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.grid()
plt.tight_layout()
plt.savefig('figure.pdf', dbi=300)
