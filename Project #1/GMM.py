import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import random

BASE_PATH = "dataset2/" # dataset2/

# Apply random data augmentation techniques to the audio.
def augment_audio(y, sr):
    # Random pitch shifting (-2 to +2 semitones)
    n_steps = random.uniform(-2, 2)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Add Gaussian noise with random variance
    noise_level = random.uniform(0.002, 0.01)
    noise = np.random.normal(0, noise_level, y.shape)
    y = y + noise

    return y

# Function to extract audio features from a given file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    y = augment_audio(y, sr) # Apply data augmentation or not, can be annotated

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Extract MFCC features
    zcr = librosa.feature.zero_crossing_rate(y) # Extract ZCR feature
    rms = librosa.feature.rms(y=y) # Extract RMS energy feature

    # Combine extracted features into a single feature vector
    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(zcr),
        np.std(zcr),
        np.mean(rms),
        np.std(rms)
    ])
    return feature_vector

# Function to calculate ACC using Hungarian Algorithm
def cluster_accuracy(y_true, y_pred):
    labels = np.unique(y_true)
    clusters = np.unique(y_pred)

    # Compute confusion matrix
    cost_matrix = np.zeros((len(labels), len(clusters)))
    for i, label in enumerate(labels):
        for j, cluster in enumerate(clusters):
            cost_matrix[i, j] = -np.sum((y_true == label) & (y_pred == cluster))  # Negative for minimization

    # Hungarian algorithm to find the best mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Remap cluster labels to match ground truth
    mapping = {clusters[col]: labels[row] for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([mapping[label] for label in y_pred])

    return accuracy_score(y_true, mapped_preds)

if __name__ == "__main__":
    # Initialize data storage
    data = []
    labels = []

    # Dataset and class labels
    base_path = BASE_PATH
    class_names = ["Drum_Solo", "Piano_Solo", "Violin_Solo", "Acoustic_Guitar_Solo", "Electric_Guitar_Solo"]
    
    # Loop through each instrument category
    for label, class_folder in enumerate(class_names):
        print(f"Extract features from {class_folder}...")
        class_path = os.path.join(base_path, class_folder)

        # Loop through each audio file in the category
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                file_path = os.path.join(class_path, file)

                # Extract features from the audio file
                features = extract_features(file_path)

                # Append features and corresponding label
                data.append(features)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    # Normalize the feature data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # GMM
    num_clusters = len(class_names)
    gmm = GaussianMixture(n_components=num_clusters, random_state=777)
    predicted_labels = gmm.fit_predict(data)

    # Calculate ARI
    ari_score = adjusted_rand_score(labels, predicted_labels)
    print(f"ARI score: {ari_score:.4f}")

    # Calculate ACC
    acc_score = cluster_accuracy(labels, predicted_labels)
    print(f"Accuracy: {acc_score:.4f}")

    # Use PCA to reduce dimension to 2D for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    # Plot
    plt.figure(figsize=(8, 6))
    unique_clusters = set(predicted_labels)
    for cluster in unique_clusters:
        plt.scatter(data_2d[predicted_labels == cluster, 0], 
                    data_2d[predicted_labels == cluster, 1], 
                    label=f'Cluster {cluster}')
    plt.title("GMM Clustering Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig("GMM.png")
