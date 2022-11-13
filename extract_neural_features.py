import numpy as np
import matplotlib.pyplot as plt
import os
import pvml

classes = os.listdir("images/test")
classes.sort()
print(classes)


def extract_neural_features(image, net):
    X = image[None, :, :, :]
    activations = net.forward(X)
    # for a in activations:
    #   print(a.shape)
    features = activations[-3].reshape(-1)
    return features


def process_directory(path, net):
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
            features = extract_neural_features(image, net)
            # print(features.shape)
            # plt.plot(range(64), features.T)
            # plt.show()
            features = features.reshape(-1)  # turn the matrix into a vector
            all_features.append(features)
            all_labels.append(class_label)
        class_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


cnn = pvml.CNN.load("pvmlnet.npz")

X, Y = process_directory("images/test", cnn)
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test2.txt.gz", data)

X, Y = process_directory("images/train", cnn)
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train2.txt.gz", data)
