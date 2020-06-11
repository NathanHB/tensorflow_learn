import tensorflow as tf
import numpy as np

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))

ndarrays = np.ones([3, 3])
print(tf.multiply(ndarrays, 43).numpy())

x = tf.random.uniform([3, 3])

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

for x in ds_tensors:
    print(x)
