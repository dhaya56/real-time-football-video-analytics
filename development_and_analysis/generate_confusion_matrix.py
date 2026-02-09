import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define the paths to your predicted and ground truth folders
predicted_dir = 'output_annotations'  # Replace with your actual directory for predicted annotations
ground_truth_dir = 'ground_truth'  # Replace with your actual directory for ground truth annotations

# List of class names (Ensure this matches your dataset classes)
class_names = ['background', 'ball', 'player', 'referee']  # Modify according to your class names

# Initialize empty lists to store ground truth and predicted labels
gt_classes = []
pred_classes = []

# Function to read the annotations from .txt files
def read_annotations(directory):
    annotations = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            frame_id = int(filename.split('_')[1].split('.txt')[0])  # Assuming filenames are like frame_0001.txt
            with open(os.path.join(directory, filename), 'r') as file:
                annotations[frame_id] = [line.split()[0] for line in file.readlines()]
    return annotations

# Read ground truth and predicted annotations
gt_annotations = read_annotations(ground_truth_dir)
pred_annotations = read_annotations(predicted_dir)

# Ensure both ground truth and predicted annotations have the same frame ids
frame_ids = sorted(set(gt_annotations.keys()) & set(pred_annotations.keys()))

# Populate the gt_classes and pred_classes lists
for frame_id in frame_ids:
    gt_classes.extend(gt_annotations[frame_id])
    pred_classes.extend(pred_annotations[frame_id])

# Convert class labels to integers (if necessary)
gt_classes = [int(cls) for cls in gt_classes]
pred_classes = [int(cls) for cls in pred_classes]

# Generate the confusion matrix
cm = confusion_matrix(gt_classes, pred_classes, labels=range(len(class_names)))

# Check if the confusion matrix is empty or not
if cm.size == 0:
    print("Confusion matrix is empty!")
else:
    # Print the confusion matrix for debugging
    print("Confusion Matrix:\n", cm)

    # Generate a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Add labels and title to the heatmap
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Display the heatmap
    plt.show()

