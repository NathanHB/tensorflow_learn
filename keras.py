from tensorflow.keras import layers
import numpy as np
import tensorflow

# ==== SETUP DATASET ====

data = np.random.random_sample((1000, 32))
labels = np.random.random_sample((1000, 10))

dataset = tensorflow.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_data = np.random.random_sample((1000, 32))
val_labels = np.random.random_sample((1000, 10))

val_dataset = tensorflow.data.Dataset.from_tensor_slices((data, labels))

# ==== CLASSIC MODEL BUILDING ====

# add layers to the model
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
# output layer
#model.add(layers.Dense(10))


# ==== FUNCTONAL MODEL BUILDING ====

#inputs = tensorflow.keras.Input(shape=(32))
#x = layers.Dense(64, activation=tensorflow.nn.relu)(inputs)
#x = layers.Dense(64, activation=tensorflow.nn.relu)(x)
#predictions = layers.Dense(10)(x)
#model = tensorflow.keras.Model(inputs=inputs, outputs=predictions)

# ==== BUILDING OWN LAYER ====
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                shape=(input_shape[1], self.output_dim),
                initializer='uniform',
                trainable=True)

    def call(self, inputs):
        return tensorflow.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==== MODEL SUBCLASSING ====

class MyModel(tensorflow.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # defining layers here
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(32, activation='relu')
        # output
        self.dense_3 = MyLayer(num_classes)

    def call(self, inputs):
        # define the forward pass
        x = self.dense_1(inputs)
        y = self.dense_2(x)
        return self.dense_3(y)


model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(0.001),
              loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tensorflow.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tensorflow.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(dataset, epochs=10, callbacks=callbacks,
        validation_data=(val_data, val_labels))
#model.summary()
