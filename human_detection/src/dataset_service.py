from src.dataset_util import *
from src.json_reader import *
from src.configuration import *
import os


def get_width_scaled(width):
    return width / WIDTH_SCALED_RATIO


def get_height_scaled(height):
    return height / HEIGHT_SCALED_RATIO


def get_tf_example(filename):
    file_path = os.path.join(filename + '.png')
    with tf.io.gfile.GFile(file_path, 'rb') as file:
        encoded_png = file.read()

    encoded_filename = filename.encode('utf8')
    image_format = b'png'
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    classes_text = []
    classes = []

    json_filename = filename + '.json'
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
