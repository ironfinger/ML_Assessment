"""
Managing Graphs
"""
#%%
import tensorflow as tf
#import tensorflow.compat.v1 as tf1
#%%
# Any node you create is added to the default graph:
x1 = tf.Variable(1)
print(x1.value())
# Dunno about this :/

# %%
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
# %%
"""Lifecycle of a Node Value"""
w = tf.constant(3)
@tf.function
def x (w): return w + 2

print(x)


# %%
