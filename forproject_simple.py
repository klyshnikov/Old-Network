import module_network as ai             #Содержит основную библиотеку
import numpy as np                       #Для вычислений
import matplotlib.pyplot as plt           #Для построения графиков

new_nework = ai.NeuralNetworkFeedForward(3, 3, 1)
new_nework.list_biases = [0.2, 0.1, -0.4, 0.1, -0.1, 0.9, 0.1]
new_nework.set_normal_values()
data = np.array(
    [[1, 3, -4], [6, 2, 4], [7, -2, 1], [9, 3, -7], [2, 6, 8], [2, 2, 2], [1, 3, 7], [5, -7, 4], [8, -9, 9], [9, 1, 2],
     [4, -4, 3], [3, 7, 6], [4, 8, 5], [99, 0, 0], [1, 0, 0], [44, 0, 0]])
all_y_trues = np.array([[0], [1], [0], [0], [1], [1], [1], [0], [0], [1], [0], [1], [1], [0], [0], [0]])

#new_nework.list_inp_hid_weights = [[-1.1479165488034204, 0.07543268454316016, 0.6139480509965898], [-1.1479165488034204, 0.07543268454316016, 0.6139480509965898], [-1.1479165488034204, 0.07543268454316016, 0.6139480509965898]]
#new_nework.list_hid_out_weights = [[0.7189091490911989, -1.3418271777568016, 0.26930446109944284]]
new_nework.epochs = 1000
new_nework.train(data, all_y_trues)

plt.plot(new_nework.x_means, new_nework.y_means)
plt.show()