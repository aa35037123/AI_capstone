from skimage.feature import hog
from glob import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

NUM_CLASSES = 7
def extract_hog_features(img_path):
    image = imread(img_path)
    image_resized = resize(image, (64, 64))  # Resize to standardize size
    fd, hog_image = hog(image_resized, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    print(f'fd size: {fd.shape}')
    return fd

def plot_class_distribution(original_sizes, resampled_sizes, title, result_root):
    classes = list(original_sizes.keys())
    n_classes = len(classes)
    index = np.arange(n_classes)
    bar_width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(index, list(original_sizes.values()), bar_width, label='Original')
    plt.bar(index + bar_width, list(resampled_sizes.values()), bar_width, label='Resampled')

    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.xticks(index + bar_width / 2, classes)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_root, f'data_distribution.jpg'))
    plt.close()

def prepare_dataset_resampling(dataset_root, result_root):
    features = []
    labels = []
    classes = sorted(os.listdir(dataset_root))
    indices_per_class = {cls: [] for cls in classes}
    
    for cls_name in classes:
        cls_index = classes.index(cls_name)
        cls_path = os.path.join(dataset_root, cls_name)
        for img_path in glob(os.path.join(cls_path, '*.jpg')):
            hog_features = extract_hog_features(img_path)
            # features.append(hog_features)
            features.append(resize(imread(img_path), (64, 64)).flatten())# Use the original image as features
            labels.append(cls_index)
            indices_per_class[cls_name].append(len(features) - 1) # record the index of the feature
    
    original_class_size = {cls: len(indices) for cls, indices in indices_per_class.items()}
    total_original_images = sum(original_class_size.values())  # Calculate total images before resampling
    min_class_size = min(len(indices) for indices in indices_per_class.values())
    total_resampled_images = min_class_size * len(classes)  # Calculate total images after resampling

    downsampled_indices = []
    for cls_indices in indices_per_class.values():
        downsampled_indices.extend(np.random.choice(cls_indices, min_class_size, replace=False))

    resampled_class_size = {cls: min_class_size for cls in classes}
    features = [features[i] for i in downsampled_indices]
    labels = [labels[i] for i in downsampled_indices]

    plot_class_distribution(original_class_size, resampled_class_size, 'Class Distribution', result_root)
    print(f'Total images before resampling: {total_original_images}')
    print(f'Total images after resampling: {total_resampled_images}')
    print(f'Data reduced: {(total_resampled_images / total_original_images):.4f}')
    # Stratified is used to ensure that each fold is a good representative of the whole dataset
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = kf.split(features, labels)
    return splits, np.array(features), np.array(labels)

def prepare_dataset(dataset_root):
    features = []
    labels = []
    classes = sorted(os.listdir(dataset_root))
    for cls_name in classes:
        cls_index = classes.index(cls_name)
        cls_path = os.path.join(dataset_root, cls_name)
        for img_path in glob(os.path.join(cls_path, '*.jpg')):
            hog_features = extract_hog_features(img_path)
            # features.append(hog_features)
            # features.append(hog_features) # Use the original image as features
            features.append(resize(imread(img_path), (64, 64)).flatten())  # Use the original image as features
            labels.append(cls_index)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = kf.split(features, labels)
    return splits, np.array(features), np.array(labels)



# def save_confusion_matrix(cm, filename):
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.savefig(filename)

def save_confusion_matrix(cm, filename, class_names):
    plt.figure(figsize=(10, 7))  # Increase figure size for better readability
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, cbar_kws={'label': 'Scale'})
    plt.title('Kfold Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # ticks specify the position of the labels(class name)
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0)
    plt.tight_layout()  # Adjust the layout
    plt.savefig(filename)
    plt.close()  # Close the figure to prevent it from displaying in Jupyter notebooks/interactive environments


def Trainer(
        kf_splits: StratifiedKFold,
        features: np.ndarray,
        labels: np.ndarray,
        result_root: str
    ) -> list[float]:
        kf_accs = []
        global_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # Assuming NUM_CLASSES is defined

        for fold, (train_index, valid_index) in enumerate(kf_splits):
            X_train, X_test = features[train_index], features[valid_index]
            y_train, y_test = labels[train_index], labels[valid_index]
            # Train the classifier
            svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, decision_function_shape='ovo'))
            svm_clf.fit(X_train, y_train)
            # Evaluate the classifier
            y_pred = svm_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            kf_accs.append(accuracy)
            cm = confusion_matrix(y_test, y_pred)
            global_cm += cm
            # Call the function inside the Trainer function
            # save_confusion_matrix(cm, os.path.join(result_root, f'confusion_fold{fold}_{accuracy}.jpg'))

    
            print(f'Fold{fold} Validation Accuracy: {accuracy:.4f}')
            # print(f'Fold{fold} Confusion Matrix:\n{cm}')
        return kf_accs, global_cm

if __name__ == '__main__':
    dataset_root = '/home/aa35037123/Wesley/ai_capstone/dataset/vehicle_merged'
    result_root = '/home/aa35037123/Wesley/ai_capstone/project1/result/svm'
    kf_splits, features, labels = prepare_dataset_resampling(dataset_root, result_root)
    # kf_splits, features, labels = prepare_dataset(dataset_root)
    kf_accs, global_cm = Trainer(kf_splits, features, labels, result_root)
    save_confusion_matrix(global_cm, os.path.join(result_root, f'final_confusion_resampling.jpg'), class_names=sorted(os.listdir(dataset_root)))


    print(f'Average accuracy: {np.mean(kf_accs):.4f}')
    
