import os
import shutil
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import tensorflow as tf

random.seed(42)

def split_data(image_dir, label_df, value):
    '''
    Split and organize image data into train, test, or validation directories based on given labels.

    This function takes in image data and corresponding label information in the form of a DataFrame,
    and organizes the images into separate directories for training, testing, or validation purposes.

    Args:
    -----------------
    - image_dir (str): Path to the directory containing the image files.
    - label_df (pandas.DataFrame): DataFrame containing label information, including filenames and species labels.
    - The DataFrame must have a "Species" column.
        value (str): Type of data split, either 'train', 'test', or 'validation'.

    Returns:
    -----------------
        None

    Raises:
    -----------------
        ValueError: If label_df is not a pandas DataFrame or if it doesn't contain a "Species" column.
    '''

    if not isinstance(label_df, pd.DataFrame):
        raise ValueError("The 'label_df' argument must be a pandas DataFrame.")

    if "Species" not in label_df.columns:
        raise ValueError("The 'label_df' DataFrame must contain a 'Species' column.")

    for ind, row in label_df.iterrows():
        # Check if the target directory exists, if not create it
        if not os.path.exists(value):
            os.mkdir(value)
            os.mkdir(f'{value}/{row["Species"]}')
            shutil.move(f'{image_dir}/{row["filename"]}', f'{value}/{row["Species"]}')
        else:
            # If the target species directory doesn't exist, create it
            if not os.path.exists(f'{value}/{row["Species"]}'):
                os.mkdir(f'{value}/{row["Species"]}')
                shutil.move(f'{image_dir}/{row["filename"]}', f'{value}/{row["Species"]}')
            else:
                # Copy the image file to the corresponding species directory
                shutil.move(f'{image_dir}/{row["filename"]}', f'{value}/{row["Species"]}')


def extract_images_and_labels(image_tensors, normalize=True, categorical=True):
    """
    Extracts images and labels from a list of image tensors and their corresponding labels.

    This function takes a list of image tensors along with their corresponding labels,
    extracts the images and labels, and optionally normalizes the images.

    Args:
        image_tensors (list of tuples): A list of tuples where each tuple contains an image tensor and its label tensor.
        normalize (bool, optional): Flag indicating whether to normalize the images by dividing by 255.0. Default is True.

    Returns:
        tuple: A tuple containing two numpy arrays - images and labels.
            images (numpy.ndarray): An array containing the extracted image data.
            labels (numpy.ndarray): An array containing the corresponding label data.

    Example:
        image_tensors = [(image_tensor_1, label_tensor_1), (image_tensor_2, label_tensor_2), ...]
        images, labels = extract_images_and_labels(image_tensors)
    """
    images = []
    labels = []
    
    for image, label in image_tensors:
        images.append(image.numpy()[0])  # Extract the image data from the tensor
        labels.append(label.numpy()[0])  # Extract the label data from the tensor
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize the images
    if normalize:
        images = images / 255.0
    if categorical:
        labels = tf.keras.utils.to_categorical(labels)
    
    return images, labels


def visualize_dimensionality_reduction(train_images, train_labels, labels, perplexity=5, n_components=2, random_state=42):
    '''
    Visualize dimensionality reduction techniques (PCA, t-SNE, UMAP) for image data.

    This function takes in image data and corresponding labels, and uses PCA, t-SNE, and UMAP
    to perform dimensionality reduction and visualize the results in a 1x3 grid.

    Args:
    -----------------
        images (numpy.ndarray): Array of training images.
        labels (numpy.ndarray): Array of labels corresponding to the training images.
        labels (dict): Dictionary mapping class indices to class names.

    Returns:
    -----------------
        None

    Example:
        visualize_dimensionality_reduction(train_images, train_labels, labels)
    '''

    # Flatten the train_images array
    num_samples, height, width, num_channels = train_images.shape
    flattened_train_images = train_images.reshape(num_samples, height * width * num_channels)

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(flattened_train_images)

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(flattened_train_images)

    # Perform UMAP
    umap_model = UMAP(n_components=n_components, random_state=random_state)
    umap_result = umap_model.fit_transform(flattened_train_images)

    # Plot PCA, t-SNE, and UMAP results in one row with three columns
    plt.figure(figsize=(18, 6))

    # PCA plot
    plt.subplot(1, 3, 1)
    for label in np.unique(train_labels):
        indices = np.where(train_labels == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=labels[label])  # Using labels dictionary
    plt.title('PCA')
    plt.legend()

    # t-SNE plot
    plt.subplot(1, 3, 2)
    for label in np.unique(train_labels):
        indices = np.where(train_labels == label)
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=labels[label])  # Using labels dictionary
    plt.title('t-SNE')
    plt.legend()

    # UMAP plot
    plt.subplot(1, 3, 3)
    for label in np.unique(train_labels):
        indices = np.where(train_labels == label)
        plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=labels[label])  # Using labels dictionary
    plt.title('UMAP')
    plt.legend()

    plt.tight_layout()
    plt.show()


