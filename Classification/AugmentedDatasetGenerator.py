import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#global variables definition
global healthy_angle
healthy_angle = 0.2893632714757443
global unhealthy_angle
unhealthy_angle = 0.010052845314350523
global healthy_shear
healthy_shear = 44.92674427780217
global unhealthy_shear
unhealthy_shear = 40.27315507780925
global healthy_horizontal_shift
healthy_horizontal_shift = 0.5
global unhealthy_horizontal_shift
unhealthy_horizontal_shift = 0.4125690647612485
global healthy_vertical_shift
healthy_vertical_shift = 0.3059300917515268
global unhealthy_vertical_shift
unhealthy_vertical_shift = 0.42491135655726076
global healthy_zoom
healthy_zoom = 0.38851838081044565
global unhealthy_zoom
unhealthy_zoom = 0.02615242144834687
global healthy_brightness_upper
healthy_brightness_upper= 0.1301597604699279
global unhealthy_brightness_upper
unhealthy_brightness_upper= 0.1301597604699279
global healthy_brightness_lower
healthy_brightness_lower = 0.3898007820036226
global unhealthy_brightness_lower
unhealthy_brightness_lower = 0.3898007820036226
global healthy_constrast
healthy_constrast = 0.8931125766720235
global unhealthy_constrast
unhealthy_constrast = 0.9283129552794223
global healthy_color_invert
healthy_color_invert = 0.5
global unhealty_color_invert
unhealty_color_invert = 0.5
global healthy_horizontal_flip
healthy_horizontal_flip = 0.5
global unhealthy_horizontal_flip
unhealthy_horizontal_flip = 0.5
global healthy_vertical_flip
healthy_vertical_flip = 0.5
global unhealthy_vertical_flip
unhealthy_vertical_flip = 0.5
global seed
seed = 69
global smallnumber
smallnumber = 0.0000000000000001

global transformations_light
transformations_light = [0, 0, 0]
global transformations_heavy
transformations_heavy = [0, 0, 0, 0, 0]


def standardize(X):
    min_value = X.min()
    max_value = X.max()

    max_value = max_value - min_value

    X = (((X - min_value) / max_value) * 255).astype(int)
    return X


class InspectImages:
    def __init__(self, path):
        self.path = path
        # load the data from a npz file
        self.data = np.load(self.path, allow_pickle=True)
        # get the images and labels
        self.images = self.data['data']
        self.images = standardize(self.images)
        self.labels = []  # target values
        for i in range(len(self.data['labels'])):
            self.labels.append((self.data['labels'])[i])
        self.labels = np.array(self.labels)
        # get the number of images
        self.num_images = self.images.shape[0]
        # get the number of classes
        self.num_classes = self.labels.shape[0]
        # get the image size
        self.image_size = self.images.shape[1]
        # get the image channels
        self.image_channels = self.images.shape[3]
        # get the number of transformations
        self.num_transformations = 8
        self.transformations = np.zeros(8)
        self.num_healthy = 0
        self.num_unhealthy = 0

    def get_images(self):
        images = self.images
        return images

    def get_labels(self):
        labels = self.labels
        return labels

    def del_image(self, index):
        # delete the image
        self.images = np.delete(self.images, index, axis=0)  # the function natively adjusts the indexes
        # delete the label
        self.labels = np.delete(self.labels, index, axis=0)
        # update the number of images
        self.num_images = self.images.shape[0]
        return self.images, self.labels

    def custom_save_to_npz(self, title):
        # create a dictionary with the data
        data = {'data': self.images, 'labels': self.labels}
        # save the data as a npz file
        np.savez(title, **data)

    def count_healthy_and_unhealthy(self):
        for i in range(self.images.shape[0]):
            if self.labels[i] == 'healthy':
                self.num_healthy += 1
        for i in range(self.images.shape[0]):
            if self.labels[i] == 'unhealthy':
                self.num_unhealthy += 1
        print('There are ', self.num_healthy, 'healthy images')
        print('There are ', self.num_unhealthy, 'unhealthy images')

    def augment_all_images_multiple(self):
        for i in range(self.images.shape[0]):
            if i % 500 == 0:
                print(str(i), 'images augmented')
            self.images[i] = self.augment_image_multiple(self.images[i],self.labels[i])

    def flip_all_images(self):
        flipped_healthy = 0
        # Cycle all the images inside the class object
        for i in range(self.images.shape[0]):

            # Print a process update every 500 images
            if i % 500 == 0:
                print(str(i), 'images flipped')

            if self.labels[i] == 'healthy':
                # If the image is healthy we flip it randomly in vertical or horizontal
                choice = np.random.randint(0, 2)
                if choice == 0:
                    self.images[i] = self.vertical_flip_image(self.images[i], self.labels[i])
                    flipped_healthy += 1

                elif choice == 1:
                    self.images[i] = self.horizontal_flip_image(self.images[i], self.labels[i])
                    flipped_healthy += 1

            elif self.labels[i] == 'unhealthy':
                # If the image is unhealthy we substitute the native image with a vertical flipped one
                # and we append at the end the horizontal flipped version
                vertical_flipped = self.vertical_flip_image(self.images[i], self.labels[i])
                horizontal_flipped = self.horizontal_flip_image(self.images[i], self.labels[i])
                self.images[i] = horizontal_flipped
                self.images = np.append(self.images, [vertical_flipped], axis=0)
                self.labels = np.append(self.labels, [self.labels[i]], axis=0)

        # Print the shapes to inspect the dimensions at the enD
        print(self.images.shape)
        print(self.labels.shape)

    def augment_image_multiple(self, image, label):
        # create a transformation dictionary
        light_transf_dict = {'shear': 0, 'horizontal shift': 1, 'vertical shift': 2}
        heavy_transf_dict = {'brightness_decrease': 0, 'contrast': 1, 'grayscale': 2, 'blur': 3, 'green channel removal': 4}

        # select a transformation from the dictionary randomly
        select_transf = (np.random.randint(0, len(light_transf_dict)), np.random.randint(0, len(heavy_transf_dict)))

        # return the augmented image and update the counter

        # shear
        if select_transf[0] == light_transf_dict['shear']:
            transformations_light[0] += 1
            image = self.shear_image(image, label)
        # horizontal shift
        elif select_transf[0] == light_transf_dict['horizontal shift']:
            transformations_light[1] += 1
            image = self.horizontal_shift_image(image, label)
        # vertical shift
        elif select_transf[0] == light_transf_dict['vertical shift']:
            transformations_light[2] += 1
            image = self.vertical_shift_image(image, label)

        # brightness down
        if select_transf[1] == heavy_transf_dict['brightness_decrease']:
            transformations_heavy[0] += 1
            image = self.brightness_image_decrease(image, label)
        # contrast
        elif select_transf[1] == heavy_transf_dict['contrast']:
            transformations_heavy[1] += 1
            image = self.contrast_image(image, label)
        # grayscale
        elif select_transf[1] == heavy_transf_dict['grayscale']:
            transformations_heavy[2] += 1
            image = self.grayscale_image(image, label)
        # blur
        elif select_transf[1] == heavy_transf_dict['blur']:
            transformations_heavy[3] += 1
            image = self.blur_image(image, label)
        # green channel removal
        elif select_transf[1] == heavy_transf_dict['green channel removal']:
            transformations_heavy[4] += 1
            image = self.green_channel_removal_image(image, label)

        return image

    def shear_image(self, image, label):
        # Select the shear angle depending on the optimized value
        if label == 'healthy':
            phi = healthy_shear
        elif label == 'unhealthy':
            phi = unhealthy_shear

        # Apply the transformation
        image = tf.keras.preprocessing.image.random_shear(image, phi, row_axis=0, col_axis= 1, channel_axis= 2)

        # Return the image
        return image

    def horizontal_flip_image(self, image, label):
        # Apply the transformation
        image = tf.image.flip_left_right(image)

        # Return the image
        return image

    def vertical_flip_image(self, image, label):
        # Apply the transformation
        image = tf.image.flip_up_down(image)

        # return the image
        return image

    def brightness_image_decrease(self, image, label):
        # Select the brightness factor depending on the optimized value
        if label == 'healthy':
          phi = healthy_brightness_upper
        elif label == 'unhealthy':
          phi = unhealthy_brightness_upper

        # change the brightness
        image = ((tf.keras.layers.RandomBrightness((-phi, -0.9*phi))(image)).numpy()).astype(int)

        # return the image
        return image

    def contrast_image(self, image, label):
        # Select the contrast factor depending on the optimized value
        if label == 'healthy':
          phi = healthy_constrast
        elif label == 'unhealthy':
          phi = unhealthy_constrast

        # Apply the transformation
        image = (tf.keras.layers.RandomContrast(phi,phi))(image)

        # Return the image
        return image

    def grayscale_image(self, image, label):
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

        # add the third dimension = 3 such that is 96, 96, 3
        image = np.stack((image,) * 3, axis=-1)
        image = image.astype(int)

        # return the image
        return image

    def blur_image(self, image,label):
        # blur the image
        image = gaussian_filter(image, sigma=0.7)

        # return the image
        return image

    def horizontal_shift_image(self, image, label):
        # Select the shift factor depending on the optimized value
        if label == 'unhealthy':
            phi = healthy_horizontal_shift
        elif label == 'healthy':
            phi = unhealthy_horizontal_shift

        # Apply the transformation
        image = tf.keras.layers.RandomTranslation((-phi, phi), 0, 'wrap')(image)

        # Return the image
        return image

    def vertical_shift_image(self, image, label):
        # Select the brightness factor depending on the optimized value
        if label == 'unhealthy':
            phi = unhealthy_vertical_shift
        elif label == 'healthy':
            phi = healthy_vertical_shift

        # Apply the transformation
        image = tf.keras.layers.RandomTranslation(0, (-phi, phi), 'wrap')(image)

        # Return the image
        return image

    def green_channel_removal_image(self, image, label):
        # Copy the image
        image1= np.copy(image)

        #Set to 0 the green channel
        image1[:, :, 1] = 0

        # Return the image
        return image

if __name__ == '__main__':
    print('start')

    # Load native dataset
    dataset = np.load('public_data_no_outliers.npz', allow_pickle=True)

    # Create the class object with native data
    mydata = InspectImages('public_data_no_outliers.npz')

    print('Number of images: ', mydata.images.shape[0])

    # Get the names of the files in a folder
    dir_name = 'duplicated_images'
    files = os.listdir(dir_name)

    # Get the last part of the name
    indexes = []
    for file in files:
        name = file.split('_')[-1]
        name = name.split('.')[0]
        indexes.append(name)

    # Cast all the values of the arrays
    for i in range(indexes.__len__()):
        indexes[i] = int(indexes[i])

    # Reverse sort the indexes to avoid missmatch
    indexes = sorted(indexes, reverse=True)

    # Delete the duplicated images
    for index in indexes:
        mydata.del_image(int(index))

    print('Number of images after removing duplicates: ', mydata.images.shape[0])

    # Save images and labels
    images = mydata.get_images()
    labels = mydata.get_labels()

    # Split between training-validation and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(images, labels, random_state=seed, test_size=0.1,
                                                              stratify=labels)
    # Split between training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=seed, test_size=0.2,
                                                      stratify=y_trainval)

    # Save all the datasets
    np.savez('train_dataset_no_duplicate.npz', data=X_train, labels=y_train)
    np.savez('val_dataset.npz', data=X_val, labels=y_val)
    np.savez('test_dataset.npz', data=X_test, labels=y_test)

    # Create the class object with the new test data
    images_inspector = InspectImages('train_dataset_no_duplicate.npz')

    # print the number of images before augmenting
    print("Number of images before augmentation:", images_inspector.images.shape[0])

    # Augment the dataset
    print('Flipping the images...')
    images_inspector.flip_all_images()

    # Print and updates the number of healthy and unhealthy records
    images_inspector.count_healthy_and_unhealthy()

    print('Augmenting the dataset...')
    images_inspector.augment_all_images_multiple()

    # print the number of images after the augmentation
    print("Number of images after augmentation:", images_inspector.images.shape[0])

    print('Saving the augmented datas...')
    images_inspector.custom_save_to_npz('train_dataset_only_augmented_1')

    # print the number of transformations
    print("transformations light", transformations_light)
    print("transformations heavy", transformations_heavy)

    print('Uploading the native dataset without outliers and duplicates...')
    images_inspector_2 = InspectImages('train_dataset_no_duplicate.npz')

    # merge the two datasets
    print('Merging the native and augmented datasets...')
    images_inspector_2.images = np.append(images_inspector_2.images, images_inspector.images, axis=0)
    images_inspector_2.labels = np.append(images_inspector_2.labels, images_inspector.labels, axis=0)

    # save the merged dataset
    print('Saving the merged dataset...')
    images_inspector_2.custom_save_to_npz('train_dataset_merged_1.npz')

    # Create the class object with the merged dataset
    images_inspector_def = InspectImages('train_dataset_merged_1.npz')

    # Print the shapes of the merged dataset
    print(images_inspector_def.labels.shape)
    print(images_inspector_def.images.shape)

    # Load the new merged dataset to perform a shuffle. This should not be necessary, but we do this as a precaution

    dataset = np.load('train_dataset_merged_1.npz', allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']

    # Create a random index vector with the same length of the dataset
    num_samples = len(data)
    random_index = np.random.permutation(num_samples)

    # Sort the dataset following the new index
    shuffled_data = data[random_index]
    shuffled_labels = labels[random_index]

    # Save the shuffled dataset
    shuffled_dataset = {
        'data': shuffled_data,
        'labels': shuffled_labels
    }
    np.savez('train_dataset_aug_mix_1.npz', **shuffled_dataset)

    print("end")



    ##################################################################
    ###  This is an extra snippet to check graphically the result  ###
    ##################################################################
    #
    # # Create a matplotlib figure
    # fig, axes = plt.subplots(6, 7, figsize=(12, 10))
    #
    # # Select the images to be plotted
    # for i, ax in enumerate(axes.flat):
    #     # Assicurati che 'images' contenga le immagini in un formato appropriato
    #     image = images_inspector_def.get_image(i+600)
    #     # Plotta l'immagine
    #     ax.imshow(image)
    #     ax.axis('off')
    #     ax.set_title(images_inspector_def.data['labels'][i+600])
    #
    # # Adjust the layout
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    #
    # # Show the plot
    # plt.show()
