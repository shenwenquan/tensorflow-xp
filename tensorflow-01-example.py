import tensorflow as tf
import numpy as np

# create train data
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

# create tensorflow structure start #
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 随机数列方式生成变量
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

# 预测y值初始会和真实值相差很大，神精网络每次提升预测y的准确度（Weigths 和 biases）
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tensorflow structure end #

sess = tf.Session()
sess.run(init)  # 激活神经网络

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
