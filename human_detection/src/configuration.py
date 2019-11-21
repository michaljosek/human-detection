import os


SCALED_IMAGE_HEIGHT = 50
SCALED_IMAGE_WIDTH = 50

ORIGINAL_IMAGE_HEIGHT = 1024
ORIGINAL_IMAGE_WIDTH = 1920

HEIGHT_SCALED_RATIO = ORIGINAL_IMAGE_HEIGHT / SCALED_IMAGE_HEIGHT
WIDTH_SCALED_RATIO = ORIGINAL_IMAGE_WIDTH / SCALED_IMAGE_WIDTH

src_folder_path = os.path.dirname(os.path.abspath(__file__))
labels_folder_path = r'{0}\\resources\\roma_labels\\'.format(src_folder_path)
images_folder_path = r'{0}\\resources\\roma_images\\'.format(src_folder_path)
