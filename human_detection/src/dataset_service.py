from src.dataset_util import *
from src.json_reader import *
from src.configuration import *
import cv2
import numpy as np


def get_width_scaled(width):
    return width / WIDTH_SCALED_RATIO


def get_height_scaled(height):
    return height / HEIGHT_SCALED_RATIO


def get_tf_example(filename):
    resized_image = get_resized_image(filename)
    encoded_png = resized_image.tostring()

    encoded_filename = filename.encode('utf8')
    image_format = b'png'
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    classes_text = []
    classes = []

    json_filename = get_json_file(filename)
    labels_json = read_json(json_filename)

    for label in labels_json["children"]:
        x_mins.append(get_width_scaled(label["x0"]))
        x_maxs.append(get_width_scaled(label["x1"]))
        y_mins.append(get_height_scaled(label["y0"]))
        y_maxs.append(get_height_scaled(label["y1"]))
        classes_text.append('human'.encode('utf8'))
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(SCALED_IMAGE_HEIGHT),
        'image/width': int64_feature(SCALED_IMAGE_WIDTH),
        'image/filename': bytes_feature(encoded_filename),
        'image/source_id': bytes_feature(encoded_filename),
        'image/encoded': bytes_feature(encoded_png),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(x_mins),
        'image/object/bbox/xmax': float_list_feature(x_maxs),
        'image/object/bbox/ymin': float_list_feature(y_mins),
        'image/object/bbox/ymax': float_list_feature(y_maxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def get_resized_image(filename, size=(SCALED_IMAGE_HEIGHT, SCALED_IMAGE_WIDTH)):
    img = cv2.imread(get_png_file(filename))
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def get_tf_record(filenames, filename_output):
    with tf.io.TFRecordWriter(filename_output) as writer:
        for filename in filenames:
            tf_example = get_tf_example(filename)
            serialized = tf_example.SerializeToString()

            writer.write(serialized)


def get_json_file(filename):
    return filename + '.json'


def get_png_file(filename):
    return filename + '.png'


