import os
import shutil
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import tensorflow as tf
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec 
from PIL import Image
from sklearn.metrics import classification_report

random.seed(42)
transfer_weights = os.path.join(Path(os.getcwd()).parent, 'models', 'weights', 'MobileNetV1.0_2.96x96.color.bsize_96.lr_0_05.epoch_170.val_loss_3.61.val_accuracy_0.27.hdf5')
labels_dict = {
                0: 'BUSHBUCK',
                1: 'IMPALA',
                2: 'MONKEY',
                3: 'WARTHOG',
                4: 'WATERBUCK',
                5: 'ZEBRA'
            }

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
        labels = tf.keras.utils.to_categorical(labels, num_classes=6)
    
    return images, labels

# visualize batch of images
def visualize_images(dataloader, batch_size=6):
    # get the class names
    class_names = list(dataloader.class_names)
    # set a matplotlib figure
    plt.figure(figsize=(10, 10))
    # take the first batch of images
    for images, labels in dataloader.take(1):
        for i in range(batch_size):
            ax = plt.subplot(batch_size//3, 3, i + 1)
            try:
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
            except:
                break

# get the data distribution
def data_distribution(train_dir, val_dir, test_dir, plot=False):
    train_length = dict()
    val_length = dict()
    test_length = dict()
    
    # populate the dictionaries
    for folder in os.listdir(train_dir):
        train_length[folder] = len(os.listdir(f'{train_dir}/{folder}'))
    for folder in os.listdir(val_dir):
        val_length[folder] = len(os.listdir(f'{val_dir}/{folder}'))
    for folder in os.listdir(test_dir):
        test_length[folder] = len(os.listdir(f'{test_dir}/{folder}'))
    
    if plot:
        classes = list(train_length.keys())
        train_counts = list(train_length.values())
        test_counts = list(test_length.values())
        val_counts = list(val_length.values())
        
        width = 0.2
        x = range(len(classes))
        
        plt.bar(x, train_counts, width, label='Train')
        plt.bar([i + width for i in x], test_counts, width, label='Test')
        plt.bar([i + 2 * width for i in x], val_counts, width, label='Validation')
        
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title('Data Distribution by Class')
        plt.xticks([i + width for i in x], classes)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        return train_length, val_length, test_length

def visualize_dimensionality_reduction(train_images, train_labels, labels, perplexity=5, n_components=2, random_state=42):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP
    from sklearn.cluster import KMeans

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
    
    kmeans = KMeans(n_clusters=6, random_state=random_state)
    train_labels = kmeans.fit_predict(flattened_train_images)

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


# Model Architecture leveraging pretrained weights
def build_model(num_classes=6, dropout=0.1, activation='softmax', 
                input_shape=(96,96,3), weights=transfer_weights, alpha=0.2, 
                train_able=False, optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'], learning_rate=0.005, weight_decay=0.0001,
                momentum=0.9):
    
    """
    Build a custom classification model based on MobileNet architecture with pretrained weights.

    Parameters:
    -----------------
    - num_classes (int): Number of output classes.
    - dropout (float): Dropout rate to apply after the GlobalAveragePooling2D layer.
    - activation (str): Activation function for the output layer.
    - input_shape (tuple): Input shape of the images (height, width, channels).
    - weights (str): Pretrained weights to use. Set to 'imagenet' for ImageNet weights.
    - alpha (float): Width multiplier for the MobileNet architecture.
    - train_able (bool): Whether to make the base model trainable.
    - optimizer (str): Optimizer for model compilation ('adam' or 'sgd').
    - loss (str): Loss function for model compilation.
    - metrics (list): List of metrics for model compilation.
    - learning_rate (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for the optimizer.
    - momentum (float): Momentum for the optimizer (used if optimizer is 'sgd').

    Returns:
    -----------------
    - tf.keras.Model: Compiled classification model.
    """

    base_model = tf.keras.applications.MobileNet(input_shape=input_shape,
                                                 alpha=alpha,
                                                 weights=weights)
    base_model.trainable = train_able
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='x_input'))
    # Don't include the base model's top layers
    last_layer_index = -5
    model.add(tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
    model.add(tf.keras.layers.GlobalAveragePooling2D())  # Global average pooling
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation=activation))

    # compiling the model and defining loss function
    if optimizer == 'adam':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                         weight_decay=weight_decay),
                        loss=loss,
                        metrics=metrics)
    elif optimizer == 'sgd':
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                                        momentum=momentum,
                                                        weight_decay=weight_decay),
                        loss=loss,
                        metrics=metrics)
    

    return model

# Load random images from a directory
def inference_model(keras_model, dir='test', mapping=labels_dict):
     # Choose a random class folder
    folder = random.choice(os.listdir(dir))
    print(f'Class: {folder}')

    # Get a random image from the chosen class folder
    image = random.choice(os.listdir(f'{dir}/{folder}'))

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(f'{dir}/{folder}/{image}', 
                                                  target_size=(96, 96))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = keras_model.predict(image)

    # Get class probabilities
    probabilities = tf.nn.softmax(prediction[0]).numpy()

    # Get predicted class labels
    predicted_class = [mapping[i] for i in range(len(probabilities))]

    # Sort the probabilities
    probabilities, predicted_class = zip(*sorted(zip(probabilities, predicted_class)))

    # Create a 1x2 grid layout
    gs = GridSpec(1, 2, width_ratios=[1, 2])

    # Plot the image on the left side of the grid
    plt.subplot(gs[0, 0])
    plt.imshow(image_array)
    plt.title(f'{folder}')

    # Plot the horizontal bar chart on the right side of the grid
    plt.subplot(gs[0, 1])
    plt.barh(predicted_class, probabilities, color='purple')
    plt.xlabel('Predicted Probabilities')
    plt.title('Predicted Class Probabilities')

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# Classification report plots
def plot_classification_report(model, test_tensors, labels=labels_dict):
    test_images, test_labels = extract_images_and_labels(test_tensors, normalize=True, categorical=False)
    y_pred = model.predict(test_images)
    y_pred_bool = np.argmax(y_pred, axis=1)

    # generate classification report
    report = classification_report(test_labels, y_pred_bool, output_dict=True)
    print(report)

    # extract class names and metric from the report
    class_names = [str(label) for label in list(report.keys())[:-3]]  # Exclude 'macro avg', 'weighted avg', etc.
    precision = []
    recall = []
    f1_score = []
    for label in class_names:
        print(f'Class: {label}')
        try:
            precision.append(report[label]['precision'])
        except KeyError:
            precision.append(0)  # Handle missing precision with a default value
            
        try:
            recall.append(report[label]['recall'])
        except KeyError:
            recall.append(0)  # Handle missing recall with a default value
            
        try:
            f1_score.append(report[label]['f1-score'])
        except KeyError:
            f1_score.append(0)  # Handle missing F1-score with a default value

    # Create a bar plot using Matplotlib
    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

    ax.set_ylabel('Scores')
    ax.set_title('Classification Report')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[int(label)] for label in class_names])  # Using custom class labels
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()




