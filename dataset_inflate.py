from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

import os

def augment_images_in_folder(folder_path, save_folder_path):
    data_aug_gen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.5, 2],
        shear_range=0.5,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=(0.1, 2),
        fill_mode='nearest'
    )

    file_list = os.listdir(folder_path)

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            img = load_img(file_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=save_folder_path, save_prefix=file_name.split('.')[0], save_format='png'):
                i += 1
                if i > 30:
                    break

# Usage example
folder_path = r'C:\Users\user\Desktop\image_all\image(2)'  # Specify the path to the folder containing input images
save_folder_path = r'C:\Users\user\Desktop\image_all\image(2)\new'  # Specify the path to the folder where augmented images will be saved
augment_images_in_folder(folder_path, save_folder_path)