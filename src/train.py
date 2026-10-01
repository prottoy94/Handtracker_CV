import csv
import math
import os

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def read_data(file_path):
    feature_lists = []
    gesture_labels = []

    data_file = open(file_path, "r", newline="")
    csv_reader = csv.reader(data_file)

    first_row = True
    for row in csv_reader:
        if len(row) == 0:
            continue

        if first_row == True and row[0] == "label":
            first_row = False
            continue

        first_row = False

        if len(row) != 43:
            print("Skipping a row with an incorrect number of values.") # Print a message indicating that a row is being skipped
            continue

        gesture_labels.append(row[0])

        feature_list = []
        for value in row[1:]:
            feature_list.append(float(value))
        feature_lists.append(feature_list)

    data_file.close()
    return feature_lists, gesture_labels


def count_labels(gesture_labels):
    label_counts = {}

    for label in gesture_labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] = label_counts[label] + 1

    return label_counts


def main():
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file_path = os.path.join(project_folder, "data", "gestures.csv")
    model_folder = os.path.join(project_folder, "models")
    model_file_path = os.path.join(model_folder, "gesture_clf.joblib")

    if os.path.exists(csv_file_path) == False:
        print("Error: The data file does not exist.")
        print("Run collect_data.py first.")
        return

    feature_lists, gesture_labels = read_data(csv_file_path)

    if len(feature_lists) == 0:
        print("Error: The data file does not contain any usable samples.")
        return

    label_counts = count_labels(gesture_labels)
    label_names = list(label_counts.keys()) # Get the list of unique gesture labels from the label counts

    if len(label_names) < 2:
        print("Error: At least two different gestures are required for training.")
        return

    for label in label_names:
        if label_counts[label] < 2:
            print("Error: Every gesture needs at least two samples.")
            print("The gesture '" + label + "' has only " + str(label_counts[label]) + " sample.")
            return

    test_sample_count = max(len(label_names), math.ceil(len(feature_lists) * 0.2))
    train_sample_count = len(feature_lists) - test_sample_count

    if train_sample_count < len(label_names):
        print("Error: There are not enough samples for a training and test set.")
        print("Collect more samples for each gesture.")
        return

    training_features, testing_features, training_labels, testing_labels = train_test_split(
        feature_lists,
        gesture_labels,
        test_size=test_sample_count,
        random_state=42,
        stratify=gesture_labels,
    )

    number_of_neighbors = min(3, len(training_features))
    classifier = KNeighborsClassifier(n_neighbors=number_of_neighbors)
    classifier.fit(training_features, training_labels)

    predicted_labels = classifier.predict(testing_features)
    accuracy = accuracy_score(testing_labels, predicted_labels)

    if os.path.exists(model_folder) == False:
        os.makedirs(model_folder)

    joblib.dump(classifier, model_file_path)

    print("Training completed.")
    print("Number of samples: " + str(len(feature_lists)))
    print("Number of gestures: " + str(len(label_names)))
    print("Test accuracy: " + str(round(accuracy * 100, 2)) + "%")
    print("Saved model to: " + model_file_path)


if __name__ == "__main__":
    main()
