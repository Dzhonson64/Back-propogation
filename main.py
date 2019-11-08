import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist
import math as m
import datetime

np.random.seed(1)  # заставим numpy выдавать одинаковые набор случайных чисел для каждого запуска программы
np.set_printoptions(suppress=True)  # выводить числа в формате 0.123 а не 1.23e-1

# В `X` находятся изображения для обучения, а в `y` значения соответственно
# `X.shape` == (60000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `y.shape` == (60000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении
X, y = loadlocal_mnist(
    images_path="train-images.idx3-ubyte",
    labels_path="train-labels.idx1-ubyte")

# В `Xt` находятся изображения для тестирования, а в `yt` значения соответственно
# `Xt.shape` == (10000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `yt.shape` == (10000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении
Xt, yt = loadlocal_mnist(
    images_path="t10k-images.idx3-ubyte",
    labels_path="t10k-labels.idx1-ubyte")

print(Xt.shape)

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1.0 / (1.0 + np.exp(-x))


def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d


X = X * (1 / 255)
Xt = Xt * (1 / 255)

# Параметры:

lr = 1  # значени на которое будет домножаться дельта на каждом шаге
batch = 60  # кол-во изображений использованное для обучения на каждом шаге
epochs = 100  # кол-во эпох. Если видно что прогресс есть, но нужно больше итераций


class MnistModel:
    def __init__(self, lr=0.1, batch=60):
        self.lr = lr
        self.batch = batch

        self.W_conv = np.random.uniform(-0.05, 0.05, (8, 3, 3))
        self.W_linear = np.random.uniform(-0.05, 0.05, (8 * 26 * 26, 10))

    def load(self, conv, linear):
        with open(conv, 'rb') as f:
            self.W_conv = np.array(pickle.load(f)).reshape((8, 3, 3))

        with open(linear, 'rb') as f:
            self.W_linear = np.array(pickle.load(f)).reshape((8 * 26 * 26, -1))

    def linear_forward(self, X):
        return np.dot(X, self.W_linear)

    def sigmoid_forward(self, X):
        return sigmoid(X)

    def relu_forward(self, X):
        X_o = X.copy()
        X_o[X < 0] = 0

        return X_o

    def convolution_forward(self, X):
        print(X.shape)
        res = np.empty((len(X), 8, 26, 26))

        # for imgId in np.arange(0, X.shape[0]):
        #     for filterItem in np.arange(self.W_conv.shape[0]):
        #         result_image = np.zeros((1, 1, 26, 26))
        #         for channelItem in np.arange(X.shape[1]):
        #             result_image[0][channelItem] = self.convolve(X[imgId, channelItem], self.W_conv[filterItem])
        #         res += result_image
        #     if (imgId % 50 == 0):
        #         print(imgId, " картинка")
        # return res

        for imgId, i in enumerate(X):
            for filterItem, j in enumerate(self.W_conv):
                res_img = np.zeros((len(i), 26, 26))
                for c in range(len(i)):
                    res_img[c] += self.convolve(i[c], j)
                res[imgId][filterItem] = res_img
                # print(res[ind][indd])
        return res

    def convolve(self, img, w):
        res = np.zeros((26, 26))
        for i in np.arange(img.shape[0]-2):
            for j in np.arange(img.shape[1]-2):
                tempMatrix = img[i:i+w.shape[0], j:j+w.shape[1]]
                res[i][j] = np.sum(tempMatrix * w)
        return res
    def forward_batch(self, X):
        conv_1 = self.convolution_forward(X)
        relu_1 = self.relu_forward(conv_1)
        flatten_1 = relu_1.reshape(-1, 5408)
        linear_1 = self.linear_forward(flatten_1)
        sigmoid_1 = self.sigmoid_forward(linear_1)

        return sigmoid_1



if __name__ == "__main__":
    model = MnistModel()
    model.load('W_conv.pickle', 'W_linear.pickle')
    lenData = 200
    tp = model.forward_batch(Xt.reshape((-1, 1, 28, 28)))

    print((np.sum(yt == np.argmax(tp, axis=1)) / len(yt)))