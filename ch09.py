import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y +2

# Actually run the graph
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    init.run()
    result = f.eval()

# Managing Graphs
x1 = tf.Variable(1)

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

# Lifecycle of a Node Value
# A variable starts its life when its initializer is run,
# and it ends when the session is closed.
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session(config=config) as sess:
    print(y.eval())
    print(z.eval())


# If you want to evaluate y and z efficiently, without evaluating w and x twice as in the
# previous code, you must ask TensorFlow to evaluate both y and z in just one graph
# run, as shown in the following code:
with tf.Session(config=config) as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)

# Linear Regression with TensorFlow
# 1. Using Normal Equation
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session(config=config) as sess:
    theta_val = theta.eval()


# 2. Using Gradient Descent
# (1) Manually Computing the Gradients
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_housing_data_plus_bias = sc.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data_plus_bias]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()


# (2) Using autodiff
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()




























