"""
Training loops
"""
#%%
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = [9, 6]

# %%

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

def f(x):
    y = x**2 + 2*x - 5
    return y

y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x, f(x), label='Ground truth') 
plt.legend()
# %%

class Model(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.random.normal,
            bias_initializer=tf.random.normal
        )
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        # For keras use 'call' != __call__
        x = x[:, tf.newaxis]
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.squeeze(x, axis=1)


# %%
model = Model(64)
# %%
plt.plot(x.numpy(), y.numpy(), '.', label='data')
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Untrained predictions')
plt.title('Before training')
plt.legend()
plt.show()
# %%
# Write a basic training loop:
variables = model.variables
optimizer = tf.optimizers.SGD(learning_rate=0.1)

for step in range (1000):
    with tf.GradientTape() as tape:
        prediction = model(x)
        error = (y-prediction)**2
        mean_error = tf.reduce_mean(error)
    gradient = tape.gradient(mean_error, variables)
    optimizer.apply_gradients(zip(gradient, variables))

    
    print(f'Mean Squared error: {mean_error.numpy():0.3f}')


# %%
plt.plot(x.numpy(),y.numpy(), '.', label="data")
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Trained predictions')
plt.title('After training')
plt.legend()
plt.show()
# %%
