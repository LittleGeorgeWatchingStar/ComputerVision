import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import itertools

def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))  # You can adjust 10000 if more is desired
    
    # Initialize an array of features, which will store the sampled descriptors
    features = np.zeros((n_image * n_each, 128))

    for i, path in enumerate(image_paths):
        # Load SIFT features from path
        descriptors = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Randomly sample n_each features from descriptors, and store them in features
        randomDescriptorsSet = np.random.permutation(descriptors)[:n_each]
        features = np.vstack((features, randomDescriptorsSet))

    # TODO: pefrom k-means clustering to cluster sampled SIFT features into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=vocab_size).fit(features)
    
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    for i, path in enumerate(image_paths):
        # Load SIFT descriptors
        descriptors = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each descriptor to the closest cluster center
        clusterIndex = kmeans.predict(descriptors)

        # TODO: Build a histogram normalized by the number of descriptors
        hist, bin_edges = np.histogram(clusterIndex, bins = vocab_size, density = True)
        image_feats[i] = hist

    return image_feats

def sample_images(ds_path, n_sample):
    print("X")
    """ Sample images from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test
    n_sample: the number of images you want to sample from the dataset.
              if None, use the entire dataset. 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.jpg.txt"))
    n_files = len(files)
    print(n_files)

    if n_sample == None:
        n_sample = n_files

    # Randomly sample from the training/testing dataset
    # Depending on the purpose, we might not need to use the entire dataset
    idx = np.random.choice(n_files, size=n_sample, replace=False)
    image_paths = np.asarray(files)[idx]
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_sample)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    return image_paths, labels

def generate_confusion_matrix(y_test, y_pred, class_names, predictor_type):
    # Compute confusion matrix
    plot.clf()
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # default
    plot.figure()
    title = predictor_type + ' Confusion matrix, without normalization'
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title=title)

    # Plot normalized confusion matrix
    plot.figure()
    title = predictor_type + ' Normalized confusion matrix'
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title=title)

    plot.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plot.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.ylabel('True labels')
    plot.xlabel('Predicted labels')
    plot.tight_layout()
