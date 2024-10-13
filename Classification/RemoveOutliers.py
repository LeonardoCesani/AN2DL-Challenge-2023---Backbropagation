import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


# Standardize the images.
def standardize(X):
    min_value = X.min()
    max_value = X.max()

    max_value = max_value - min_value

    X = (((X - min_value) / max_value) * 255).astype(int)
    return X


# Define the class to analyze the images, providing all necessary methods
class InspectImages:
    def __init__(self, path):
        self.path = path
        # Load the data from a npz file
        self.data = np.load(self.path, allow_pickle=True)
        # Get the images and labels
        self.images = self.data['data']
        self.images = standardize(self.images)
        self.labels = []  # target values
        for i in range(len(self.data['labels'])):
            self.labels.append((self.data['labels'])[i])
        self.labels = np.array(self.labels)
        # Get the number of images
        self.num_images = self.images.shape[0]
        # Get the number of classes
        self.num_classes = self.labels.shape[0]
        # Get the image size
        self.image_size = self.images.shape[1]
        # Get the image channels
        self.image_channels = self.images.shape[3]

    # Get an image.
    def get_image(self, index):
        # Get the image
        image = self.images[index]
        return image

    # Plot a certain set of images.
    def plot_images(self, indices):
        # Get the number of images
        num_images = len(indices)
        # Create the figure
        fig = plt.figure(figsize=(10, 10))
        # Plot the images
        for i, index in enumerate(indices):
            image = self.images[index]
            label = self.labels[index]
            ax = fig.add_subplot(1, num_images, i + 1)
            ax.imshow(image)
            ax.set_title(f'Label: {label}')
        plt.show()
        return fig

    # Delete an image (if a set of images have to be deleted, delete according their decreasing order of idexes).
    def del_image(self, index):
        # Delete the image
        self.images = np.delete(self.images, index, axis=0)  # The function natively adjusts the indexes
        # Delete the label
        self.labels = np.delete(self.labels, index, axis=0)
        # Update the number of images
        self.num_images = self.images.shape[0]
        return self.images, self.labels

    # Get the index of an image.
    def get_index(self, images):
        for i in range(len(self.images)):
            if np.array_equal(images, self.images[i]):
                return i

    # Get the mean of a channel for each image.
    def channel_mean(self, channel):
        # Get the mean of the channel for each image and create a vector
        mean = np.mean(self.images[:, :, :, channel], axis=(1, 2))
        return mean

    # Sort the means vector.
    def sort_means(self, mean):
        # Sort in decreasing order the means vector, preserving the indices
        sorted_means = np.argsort(mean)
        return sorted_means

    # Get the images with a certain degree of red.
    def get_red_images(self, threshold=100, difference=80):
        count = 0
        i = 0
        red_images_indexes = []
        for image in self.images:
            for column in range(self.image_size):
                for row in range(self.image_size):
                    if image[column][row][0] - image[column][row][1] > difference:
                        count += 1
            if count >= threshold:
                # append the index of the image
                red_images_indexes.append(i)
            i += 1
            count = 0
        return red_images_indexes

    # Get the images with a certain degree of blue.
    def get_blue_images(self, threshold=100, difference=80):
        count = 0
        i = 0
        red_images_indexes = []
        for image in self.images:
            for column in range(self.image_size):
                for row in range(self.image_size):
                    if image[column][row][2] - image[column][row][1] > difference:
                        count += 1
            if count >= threshold:
                # append the index of the image
                red_images_indexes.append(i)
            i += 1
            count = 0
        return red_images_indexes

    # Save a certain set of images.
    def save_images(self, dir_name, indexes):
        # create the directory if it does not exist
        os.makedirs(dir_name, exist_ok=True)

        for i in range(len(indexes)):
            image = self.get_image(indexes[i])
            # transform the image to uint8
            image = image.astype(np.uint8)
            plt.imsave(f'{dir_name}/image_{indexes[i]}.jpg', image)
        # save_array(indexes, 'Vectors', dir_name)

    # Identify duplicates
    def identify_same_images(self):
        # Identify the images with the same content
        # Create a dataframe with the images
        df = pd.DataFrame(self.images.reshape(self.num_images, -1))
        # Identify the duplicated images
        duplicated_images = df[df.duplicated()]
        # Get the indices of the duplicated images
        duplicated_images_indices = duplicated_images.index.values

        return duplicated_images_indices

    # Sort the images by similarity (i.e, if they are the same image, put their indexes closer in the array).
    def sort_by_similarity(self, indexes):
        images_array = []
        for i in indexes:
            images_array.append(self.get_image(i))

        single_images = []
        for image in images_array:
            # If the there's a correspondence between image and an element in single_images
            if len(single_images) != 0:
                if not np.any(np.array([np.array_equal(image, img) for img in single_images])):
                    # Append the image to single_images
                    single_images.append(image)
            else:
                single_images.append(image)
        return single_images

    # Save the images sorted by similarity.
    def save_with_single_image(self, single_images, indexes, dir_name):
        # create the directory if it does not exist
        os.makedirs(dir_name, exist_ok=True)

        images_array = []
        for i in indexes:
            images_array.append(self.get_image(i))

        corresp_index = 0
        for i in range(len(indexes)):
            image = self.get_image(indexes[i])
            # transform the image to uint8
            image = image.astype(np.uint8)
            for j in range(len(single_images)):
                # Check if they're the same image not index
                if np.array_equal(image, single_images[j]):
                    corresp_index = j
            plt.imsave(f'{dir_name}/{corresp_index}_image_{indexes[i]}.jpg', image)
        # save_array(indexes, 'Vectors', dir_name)

    # Delete Shrek and Troll.
    def shrek_troll(self, shrek, troll):
        for images in self.images:
            if np.array_equal(images, shrek) or np.array_equal(images, troll):
                self.del_image(self.get_index(images))

    # Transform the data to a npz file.
    def transform_to_npz(self):
        # Create a dictionary with the data
        data = {'data': self.images, 'labels': self.labels}
        # Save the data as a npz file
        np.savez('public_data_no_outliers.npz', **data)


def save_array(array, dir_name, name):
    # If the directory does not exist, raise an error
    if not os.path.exists(dir_name):
        raise ValueError(f'The directory {dir_name} does not exist')

    # Save the array as a txt file
    np.savetxt(f'{dir_name}/{name}.txt', array, fmt='%d')


def main(path):
    print('Path to the images:', path)

    images_inspector = InspectImages(path)

    # Get the "darkest" images
    print('Getting the darkest images...')
    means = images_inspector.channel_mean(1)
    means = images_inspector.sort_means(means)
    images_inspector.plot_images(means[:5])

    # Get the duplicated images
    print('Getting the duplicated images...')
    duplicated_images_indices = images_inspector.identify_same_images()
    single_images = images_inspector.sort_by_similarity(duplicated_images_indices)
    images_inspector.save_with_single_image(single_images, duplicated_images_indices, 'duplicated_images')

    # Delete outliers
    print("Deleting Shrek and Troll...")
    images_inspector.shrek_troll(single_images[0], single_images[5])

    # Get the red images
    print('Getting the reddish images...')
    red_images_indexes = images_inspector.get_red_images()
    # Get the blue images
    print('Getting the blueish images...')
    blue_images_indexes = images_inspector.get_blue_images()
    # Get the images that are both red and blue
    print('Saving the reddish images...')
    images_inspector.save_images('red_images', red_images_indexes)
    print('Saving the blueish images...')
    images_inspector.save_images('blue_images', blue_images_indexes)

    # After removing the outliers, the red and blue analysis produced only "normal" images.
    # All outliers and duplicates were removed.


if __name__ == '__main__':
    # define the path to the data
    path = 'public_data.npz'
    main(path)
