#%%
import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)
# %%
print(x+x)
# %%
print(5 * x)
# %%
x @ tf.transpose(x)
# %%
tf.concat([x, x, x], axis = 0)
# %%
tf.nn.softmax(x, axis=-1)
# %%
tf.reduce_sum(x)
# %%
# This shows if tensorflow is running on a cpu:
if tf.config.list_physical_devices('GPU'):
    print('GPU')
else:
    print('Nope')
# %%
# Normal tf.Tensor objects are immutable.
# To store model weights and other mutable states use this:
var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
# %%
var.assign_add([1, 1, 1])
# %%
# Automatic Differntiation:
x = tf.Variable(1.0)
def f(x):
    y = x**2 + 2*x - 5
    return y

with tf.GradientTape() as tape:
    y = f(x)

g_x = tape.gradient(y, x) # g(x) = dy/dx
g_x
# %%
# Graphs and functions:
@tf.function
def my_func(x):
    print('Tracing. \n')
    return tf.reduce_sum(x)

x = tf.constant([1, 2, 3])
my_func(x)
# %%
