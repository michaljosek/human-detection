input_dim = 228

import numpy as np
from PIL import Image , ImageDraw
import os
import glob

images = []
image_paths = glob.glob( 'training_images/*.jpg' )
for imagefile in image_paths:
    image = Image.open( imagefile ).resize( ( input_dim , input_dim ))
    image = np.asarray( image ) / 255.0
    images.append( image )

import xmltodict

bboxes = []
classes_raw = []
annotations_paths = glob.glob('training_images/*.xml')
for xmlfile in annotations_paths:
    x = xmltodict.parse(open(xmlfile, 'rb'))
    bndbox = x['annotation']['object']['bndbox']
    bndbox = np.array([int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])
    bndbox2 = [None] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]
    bndbox2 = np.array(bndbox2) / input_dim
    bboxes.append(bndbox2)
    classes_raw.append(x['annotation']['object']['name'])

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

boxes = np.array(bboxes)
encoder = LabelBinarizer()
classes_onehot = encoder.fit_transform(classes_raw)

Y = np.concatenate([boxes, classes_onehot], axis=1)
X = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

input_shape = ( input_dim , input_dim , 3 )
dropout_rate = 0.5
alpha = 0.2

def calculate_iou( target_boxes , pred_boxes ):
    xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = K.maximum( 0.0 , xB - xA ) * K.maximum( 0.0 , yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    return iou

def custom_loss( y_true , y_pred ):
    mse = tf.losses.mean_squared_error( y_true , y_pred )
    iou = calculate_iou( y_true , y_pred )
    return mse + ( 1 - iou )

def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred )

num_classes = 3
pred_vector_length = 4 + num_classes

model_layers = [
	keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1 ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Flatten() ,

    keras.layers.Dense( 1240 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 640 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 480 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 120 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 62 ) ,
    keras.layers.LeakyReLU( alpha=alpha ) ,

    keras.layers.Dense( pred_vector_length ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
]

model = keras.Sequential( model_layers )
model.compile(
	optimizer=keras.optimizers.Adam( lr=0.0001 ),
	loss=custom_loss,
    metrics=[ iou_metric ]
)

mkdir -v inference_images

boxes = model.predict( x_test )
for i in range( boxes.shape[0] ):
    b = boxes[ i , 0 : 4 ] * input_dim
    img = x_test[i] * 255
    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
    draw = ImageDraw.Draw( source_img )
    draw.rectangle( b , outline="black" )
    source_img.save( 'inference_images/image_{}.png'.format( i + 1 ) , 'png' )


    def calculate_avg_iou(target_boxes, pred_boxes):
        xA = np.maximum(target_boxes[..., 0], pred_boxes[..., 0])
        yA = np.maximum(target_boxes[..., 1], pred_boxes[..., 1])
        xB = np.minimum(target_boxes[..., 2], pred_boxes[..., 2])
        yB = np.minimum(target_boxes[..., 3], pred_boxes[..., 3])
        interArea = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
        boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou


    def class_accuracy(target_classes, pred_classes):
        target_classes = np.argmax(target_classes, axis=1)
        pred_classes = np.argmax(pred_classes, axis=1)
        return (target_classes == pred_classes).mean()


    target_boxes = y_test * input_dim
    pred = model.predict(x_test)
    pred_boxes = pred[..., 0: 4] * input_dim
    pred_classes = pred[..., 4:]

    iou_scores = calculate_avg_iou(target_boxes, pred_boxes)
    print('Mean IOU score {}'.format(iou_scores.mean()))

    print('Class Accuracy is {} %'.format(class_accuracy(y_test[..., 4:], pred_classes) * 100))