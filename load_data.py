import functools
import numpy as np
import tensorflow as tf

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("test.csv", TEST_DATA_URL)

label_column = "survived"
labels = [0, 1]

def make_tf_dataset(csv_filepath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
            csv_filepath,
            batch_size = 5,
            label_name = label_column,
            na_value="?",
            num_epochs=1,
            ignore_errors=True,
            **kwargs)

    return dataset

raw_training_data = make_tf_dataset(train_file_path)
raw_testing_data = make_tf_dataset(train_file_path)

