#%%
import tensorflow as tf

# %%
#Tensorflow version:
print("Tensorflow version: ", tf.__version__)
# %%
# Load the MNIST dataset:
mnist = tf.keras.datasets.mnist

# Convert sample data from int to float:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %%
# Build a tf.keras.sequential model by stacking layers:
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)
# %%
# Convert logits to probabilities:
tf.nn.softmax(predictions).numpy()
# %%
# Define a loss function