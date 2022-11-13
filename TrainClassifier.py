import numpy as np
import matplotlib.pyplot as plt
import pvml
import image_features

# we can use any classifier we took before (gaussian,...)

data = np.loadtxt("test2.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)
print(Xtest.shape, Ytest.shape)

data = np.loadtxt("train2.txt.gz")
Xtrain = data[:, :-1]
Ytrain = data[:, -1].astype(int)
print(Xtrain.shape, Ytrain.shape)

nimages = Xtrain.shape[0]
nfeatures = Xtrain.shape[1]
nclasses = Ytrain.max() + 1

# here we can change if we want to add hidden layers
#mlp = pvml.MLP([nfeatures, nclasses])
mlp = pvml.MLP([nfeatures, 50, nclasses])

epochs = 5000
batch_size = 50
lr = 0.0001

train_accs = []
test_accs = []
plt.ion()

for epoch in range(epochs):
    steps = nimages // batch_size
    mlp.train(Xtrain, Ytrain, lr, batch=batch_size, steps=steps)
    if epoch % 100 == 0:
        prediction, probs = mlp.inference(Xtrain)
        train_acc = (prediction == Ytrain).mean()
        train_accs.append(train_acc)
        prediction, probs = mlp.inference(Xtest)
        test_acc = (prediction == Ytest).mean()
        test_accs.append(test_acc)
        print(epoch, train_acc, test_acc)
        plt.clf()
        plt.plot(train_accs)
        plt.plot(test_accs)
        plt.legend(["train", "test"])
        plt.pause(0.01)
mlp.save("cakes.mlp.npz")
plt.ioff()
plt.show()

# we have to make multiple features
