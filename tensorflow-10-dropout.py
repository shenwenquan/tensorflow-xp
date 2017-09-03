# 过拟合解决办法：
# 1.增加数据
# 2.正规化:L1,L2,...
# 3.dropout regularization (随机忽略一些网络)

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, lay_name, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    tf.summary.histogram(lay_name + '/outputs', outputs)  # 要有这句
    return outputs


keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8*8
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)  # 要有这句

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("logs/train", tf.get_default_graph())
test_writer = tf.summary.FileWriter("logs/test", tf.get_default_graph())

sess.run(tf.global_variables_initializer())

for i in range(5000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
    if i % 50 == 0:
        print(sess.run(cross_entropy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1}))
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
