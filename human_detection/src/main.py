


import tensorflow as tf

SHUFFLE_BUFFER = 64
SUM_OF_ALL_DATASAMPLES = 1136
BATCH_SIZE = 32
filenames_train = ['trecord1']
EPOCHS = 10
STEPS_PER_EPOC = 10


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        "image/width": tf.io.FixedLenFeature([], tf.int64)}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image/encoded'] = tf.io.decode_raw(
        parsed_features['image/encoded'], tf.uint8)

    return parsed_features['image/encoded'], 1


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image1, label1 = iterator.get_next()

    # Bring your picture back in shape
    image1 = tf.reshape(image1, (-1, 50, 50, 3))

    # Create a one hot array for your labels
    label1 = tf.one_hot(label1, 1)

    return image1, label1


image, label = create_dataset(filenames_train)
#
train_model = tf.keras.models.Sequential()
train_model.add(tf.keras.layers.Conv2D(filters=32,
               kernel_size=(2,2),
               strides=(1,1),
               padding='same',
               input_shape=(50,50,3),
               data_format='channels_last'))
train_model.add(tf.keras.layers.Activation('relu'))
train_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                     strides=2))
train_model.add(tf.keras.layers.Conv2D(filters=64,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))
train_model.add(tf.keras.layers.Activation('relu'))
train_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                     strides=2))
train_model.add(tf.keras.layers.Flatten())
train_model.add(tf.keras.layers.Dense(64))
train_model.add(tf.keras.layers.Activation('relu'))
train_model.add(tf.keras.layers.Dropout(0.25))
train_model.add(tf.keras.layers.Dense(1))
train_model.add(tf.keras.layers.Activation('sigmoid'))
train_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train_model = tf.keras.models.Sequential()
# train_model.add(tf.keras.layers.InputLayer(input_shape=(50, 50, 3)))
# train_model.add(tf.keras.layers.Flatten())
#
# train_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#
#
# train_model.add(tf.keras.layers.Dropout(0.2))
# train_model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))

train_model.summary()

#
# train_model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(50, 50, 3)),
#   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
# print(tensor.get_shape()),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dense(1, activation=tf.nn.softmax)
# ])
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

#Compile your model
train_model.compile(optimizer='adam',
                    loss=loss,
                    metrics=['accuracy'])

train_model.fit(image, label, epochs=2,
                steps_per_epoch=1000)

