from PIL import Image
import os
import numpy as np
import random
import json

def load_data(dir_path):
    train_data = []
    train_label = []
    train_dir_path = os.path.join(dir_path, "bounding_box_train")
    train_files = os.listdir(train_dir_path)
    train_files.remove(".DS_Store")
    train_files.sort(key=lambda x: int(x[0:4]))
    for file in train_files:
        train_label.append(file.split("_")[0])
        if os.path.splitext(file)[1] == ".jpg":
            file = os.path.join(train_dir_path, file)
            image = Image.open(file)
            image = np.array(image)
            train_data.append(image)

    test_data = []
    test_label = []
    test_dir_path = os.path.join(dir_path, "bounding_box_test")
    test_files = os.listdir(test_dir_path)
    test_files.remove(".DS_Store")
    for obj in list(test_files):
        if "-1" in str(obj) or "0000" in str(obj):
            test_files.remove(obj)
    test_files.sort(key=lambda x: int(x[0:4]))
    for file in test_files:
        test_label.append(file.split("_")[0])
        if os.path.splitext(file)[1] == ".jpg":
            file = os.path.join(test_dir_path, file)
            image = Image.open(file)
            image = np.array(image)
            test_data.append(image)
    return train_data, train_label, test_data, test_label


def generate_set(data, label, length):
    data_set = []
    label_set = []
    for _ in range(length):
        a = random.randint(0, len(data) - 1)
        b = random.randint(0, len(data) - 1)
        da = np.array(data[a]) / 255.0
        db = np.array(data[b]) / 255.0
        d = np.append(da, db, axis=2)
        data_set.append(d)
        if label[a] == label[b]:
            label_set.append(1)
        else:
            label_set.append(0)

    for _ in range(length):
        a = random.randint(0, len(data) - 1)
        lower_bound = 0
        upper_bound = len(data) - 1
        for mini in range(a, 0, -1):
            if label[mini] != label[a]:
                lower_bound = mini
                break
        for maxi in range(a, len(data) - 1):
            if label[maxi] != label[a]:
                upper_bound = maxi
                break
        b = random.randint(lower_bound + 1, upper_bound - 1)
        da = np.array(data[a]) / 255.0
        db = np.array(data[b]) / 255.0
        d = np.append(da, db, axis=2)
        data_set.append(d)
        if label[a] == label[b]:
            label_set.append(1)
        else:
            label_set.append(0)

    return np.array(data_set), np.array(label_set)

