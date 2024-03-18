from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from glob import glob
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import argparse
from skimage.feature import hog

def extract_hog_features(img_path):
    image = imread(img_path)
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    fd, hog_image = hog(image_resized, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd


NUM_CLASSES = 7
def prepare_dataset_resampling(dataset_root, result_root):
    features = []
    labels = []
    class_names = sorted(os.listdir(dataset_root))
    indices_per_class = {cls: [] for cls in class_names}

    for cls_index, cls_name in enumerate(class_names):
        cls_path = os.path.join(dataset_root, cls_name)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            hog_features = extract_hog_features(img_path)
            # img = imread(img_path)
            # img_resized = resize(img, (64, 64), anti_aliasing=True)  # Resize images
            # features.append(img_resized.flatten())  # Flatten the images
            features.append(hog_features)
            labels.append(cls_index)
            indices_per_class[cls_name].append(len(features) - 1)  # record the index of the feature
    min_class_size = min(len(indices) for indices in indices_per_class.values())

    downsampled_indices = []
    for cls_indices in indices_per_class.values():
        downsampled_indices.extend(random.sample(cls_indices, min_class_size))
    
    features = [features[i] for i in downsampled_indices]
    labels = [labels[i] for i in downsampled_indices]

    return np.array(features), np.array(labels), class_names

def prepare_dataset(dataset_root, result_root):
    features = []
    labels = []
    class_names = sorted(os.listdir(dataset_root))

    for cls_index, cls_name in enumerate(class_names):
        cls_path = os.path.join(dataset_root, cls_name)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            hog_features = extract_hog_features(img_path)
            # img = imread(img_path)
            # img_resized = resize(img, (64, 64), anti_aliasing=True)  # Resize images
            # features.append(img_resized.flatten())  # Flatten the images
            features.append(hog_features)
            labels.append(cls_index)

    return np.array(features), np.array(labels), class_names

def predict_labels(cluster_labels, labels):
    cluster_to_label_map = {}
    for cluster in range(NUM_CLASSES):
        indices = [i for i,c  in enumerate(cluster_labels) if c == cluster]
        true_labels = [labels[i] for i in indices] 
        print(f'true_labels: {true_labels}')
        # returns a list of the most common label and its count, but since we only need the label
        # search original labels to get the most common label in each cluster
        most_common_label = Counter(true_labels).most_common(1)[0][0] # find truth labels in each cluster
        print(f'most common lable: {most_common_label}')
        cluster_to_label_map[cluster] = most_common_label # return label number: 0, 1, 2

    predict_labels = [cluster_to_label_map[cluster] for cluster in cluster_labels]
    
    return predict_labels


# def plot_kmeans(features, cluster_labels, class_names, result_root, predict_labels):
#     plt.figure(figsize=(10, 8))
#     colors = ['C'+str(i) for i in range(NUM_CLASSES)]

#     # Plot each cluster using different color but without assigning label yet
#     for cluster in range(NUM_CLASSES):
#         cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cluster]
#         cluster_features = features[cluster_indices]
#         plt.scatter([f[0] for f in cluster_features], 
#                     [f[1] for f in cluster_features], 
#                     color=colors[cluster])

#     # Create a label dictionary mapping cluster index to class name
#     # For this, you would need a mapping from cluster_labels to true class labels
#     label_dict = {i: class_names[i] for i in range(NUM_CLASSES)}

#     # Create custom legends
#     from matplotlib.lines import Line2D
#     custom_legends = [Line2D([0], [0], marker='o', color='w', label=label_dict[i],
#                              markerfacecolor=colors[i], markersize=10) 
#                       for i in range(NUM_CLASSES)]

#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend(handles=custom_legends)
#     plt.savefig(os.path.join(result_root, 'kmeans_resampling.jpg'), dpi=300)
#     plt.close()


def plot_kmeans(features, cluster_labels, class_names, result_root):
    plt.figure(figsize=(10, 8))
    for cluster in range(NUM_CLASSES):
        cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cluster]
        cluster_features = features[cluster_indices]
        for feature in cluster_features:
            plt.scatter(feature[0], feature[1], color='C'+str(cluster), label=class_names[cluster])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.legend()
    plt.savefig(os.path.join(result_root, 'kmeans_resampling.jpg'), dpi=300)
    plt.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True) # add help to the parser
    parser.add_argument('--use_pca', type=bool, default=False,     help="path of model pretrain weight")
    args = parser.parse_args()
    dataset_root = '/home/aa35037123/Wesley/ai_capstone/dataset/vehicle_merged'
    result_root = '/home/aa35037123/Wesley/ai_capstone/project1/result/kmeans'
    features, labels, class_names = prepare_dataset(dataset_root, result_root)
    # features, labels, class_names = prepare_dataset_resampling(dataset_root, result_root)

    # Standardize features
    scalar = StandardScaler()
    features_scaled = scalar.fit_transform(features)
    print(f'features_scaled: {features_scaled}')
    ##############################
    # if use hog feature, then comment out the following line
    # Create a pipeline with PCA and KNN
    #n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
    # pca = PCA(n_components=0.8)  # Retain 95% of variance
    # features_pca = pca.fit_transform(features_scaled)
    ##############################

    # Apply KMeans
    kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled) ## change if pca
    print(f'cluster_labels: {cluster_labels}')
    # Predict labels
    predict_labels = predict_labels(cluster_labels, labels)
    print(f'predict_labels: {predict_labels}')
    print(f'labels: {labels}')
    accuracy = accuracy_score(labels, predict_labels)
    print(f"Accuracy: {accuracy:.4f}")
    ## change if pca
    plot_kmeans(features_scaled, cluster_labels, class_names, result_root)
    # Calculate Silhouette Score
    ## change if pca
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")