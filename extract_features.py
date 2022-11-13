import numpy as np
import matplotlib.pyplot as plt
import os
import image_features

classes = os.listdir("images/test")
classes.sort()
print(classes)


def process_directory(path):
    all_features = []
    all_labels = []
    class_label = 0
    for class_ in classes:
        image_files = os.listdir(path + "/" + class_)
        for filename in image_files:
            image_path = path + "/" + class_ + "/" + filename
            image = plt.imread(image_path)
            image = image / 255
            # print(image.shape, image.dtype)
            # plt.imshow(image)
            # plt.title(image_path)
            # plt.show()
            features = image_features.color_histogram(image)
            features2 = image_features.edge_direction_histogram(image)
            features3 = image_features.cooccurrence_matrix(image)
            # print(features.shape)
            # plt.plot(range(64), features.T)
            # plt.show()
            features = features.reshape(-1)  # turn the matrix into a vector
            features2 = features.reshape(-1)
            features3 = features.reshape(-1)
            newfeatures = np.concatenate((features, features2,features3))
            all_features.append(newfeatures)
            all_labels.append(class_label)
        class_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


X, Y = process_directory("images/test")
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)

X, Y = process_directory("images/train")
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)
