import module_network as ai
import numpy as np
import matplotlib.pyplot as plt

new_network = ai.NeuralNetworkFeedForward(2, 2, 1)
new_network.list_biases = [0, 0, 0, 0, 0, 0, 0]
new_network.set_normal_values()
data = np.array([[40, 11], [60, 23], [40, 27], [20, 11], [20, 12], [40, 23], [20, 12],
                 [40, 24], [60, 23], [20, 15], [20, 11], [40, 21], [60, 23], [40, 16], [20, 24], [20, 24],
                 [40, 11], [40, 11], [20, 12], [20, 26], [40, 11], [20, 12], [40, 11], [20, 12], [40, 23],
                 [40, 25], [60, 30]])
all_y_trues = np.array([[0],
                        [1], [1], [0], [1], [1], [1], [1], [1], [0], [0], [1], [1], [1], [0], [0], [0], [0], [0],
                        [0], [0], [0], [0], [0], [1], [1], [1]])


new_network.list_inp_hid_weights = np.array(
    [[0.800024785817727, -2.6359497300326415], [0.800024785817727, -2.6359497300326415]])
new_network.list_hid_out_weights = np.array([[-2.998268278590814, -2.5167838555772466]])

def predcit(a):
    a = new_network.feedforward_output(a)
    if a[0] < 0.4999997755728009:
        print("Ноутбук, скорее всего, стоит меньше 40000 рублей")
    elif a[0] > 0.4999997755728009:
        print("Скорее всего, ноутбук стоит больше 40000 рублей")
    elif a[0] == 0.4999997755728009:
        print("Стоимость ноутбука - в районе 40000 рублей")


print("+++++++++++++++++++")
predcit([20, 15])
print('Верно: меньше')
predcit([40, 24])
print('Верно: больше /+')
predcit([40, 15])
print('Верно: меньше /+')
predcit([60, 20])
print('Верно: больше')
predcit([60, 30])
print('Верно: больше /+')
predcit([40, 11])
print('Верно: меньше /+')
predcit([20, 12])
print('Верно: меньше /+')
predcit([20, 30])
print('Верно: больше /+')

print("\nКачесво сети 6/8 -> 75% правильных ответов\n")


while True:
    inp1, inp2 = input().split()
    inp = [int(inp1), int(inp2)]
    print(inp)
    predcit(inp)
    print("\n")
