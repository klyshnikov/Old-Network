import numpy as np
import matplotlib.pyplot as plt

y_means = []
x_means = []


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 10000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                y_pred = o1

                p_L_p_w1 = -2 * (y_true - y_pred) * self.w5 * x[0]
                p_L_p_w2 = -2 * (y_true - y_pred) * self.w5 * x[1]
                p_L_p_w3 = -2 * (y_true - y_pred) * self.w6 * x[0]
                p_L_p_w4 = -2 * (y_true - y_pred) * self.w6 * x[1]
                p_L_p_w5 = -2 * (y_true - y_pred) * h1
                p_L_p_w6 = -2 * (y_true - y_pred) * h2
                p_L_p_b1 = -2 * (y_true - y_pred) * self.w5
                p_L_p_b2 = -2 * (y_true - y_true) * self.w6
                p_L_p_b3 = -2 * (y_true - y_pred)

                self.w1 -= p_L_p_w1
                self.w2 -= p_L_p_w2
                self.w3 -= p_L_p_w3
                self.w4 -= p_L_p_w4
                self.w5 -= p_L_p_w5
                self.w6 -= p_L_p_w6
                self.b1 -= p_L_p_b1
                self.b2 -= p_L_p_b2
                self.b3 -= p_L_p_b3

            print(self.feedforward(data))
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            x_means.append(epoch)
            y_means.append(loss)
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


data = np.array([
    [0, 0],  # 1
    [0, 1],
    [1, 0],  # 3
    [1, 1],
    [3, 1],  # 5
    [1, 3],
    [4, 0],  # 7
    [0, 4],
    [2, 9],  # 9
    [9, 2],
    [6, 4],  # 11
    [4, 6],
    [5, 5],  # 13
    [3, 3],
])

all_y_trues = np.array([
    0,  # 1
    1,
    1,  # 3
    2,
    4,  # 5
    4,
    4,  # 7
    4,
    11,  # 9
    11,
    10,  # 11
    10,
    10,  # 13
    6,
])

network = OurNeuralNetwork()
network.train(data, all_y_trues)

five = np.array([3, 2])
print("Emily: %.3f" % network.feedforward(five))

plt.plot(x_means, y_means)
plt.show()
