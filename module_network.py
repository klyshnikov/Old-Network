import numpy as np
import matplotlib.pyplot as plt

activation_functions = ["SIGMOID", "TANH", "RELU"]


class NeuralNetworkFeedForward():

    def __init__(self, deg_input_neurons: int, deg_hidden_neurons: int, deg_output_neurons: int):
        self.deg_input_neurons = deg_input_neurons
        self.deg_hidden_neurons = deg_hidden_neurons
        self.deg_output_neurons = deg_output_neurons

        self.list_inp_hid_weights = [[None] * self.deg_input_neurons] * self.deg_hidden_neurons
        self.list_hid_out_weights = [[None] * self.deg_hidden_neurons] * self.deg_output_neurons
        self.list_biases = [None] * (deg_input_neurons + deg_hidden_neurons + deg_output_neurons)

        self.activation_f = self.sigmoid
        self.list_const_values_0 = []
        self.list_const_values_1 = []
        self.list_const_biases = []
        self.learn_rate = 0.3
        self.epochs = 10000

        self.x_means = []
        self.y_means = []

    # ===================SETS FUNCTIONS==================

    def set_normal_values(self):
        for i1 in range(self.deg_hidden_neurons):
            for i2 in range(self.deg_input_neurons):
                self.list_inp_hid_weights[i1][i2] = np.random.normal()
        for i1 in range(self.deg_output_neurons):
            for i2 in range(self.deg_hidden_neurons):
                self.list_hid_out_weights[i1][i2] = np.random.normal()

    def set_random_values(self, min_value, max_value):
        for i1 in range(self.deg_hidden_neurons):
            for i2 in range(self.deg_input_neurons):
                self.list_inp_hid_weights[i1][i2] = np.random.randint(min_value, max_value)
        for i1 in range(self.deg_output_neurons):
            for i2 in range(self.deg_hidden_neurons):
                self.list_hid_out_weights[i1][i2] = np.random.randint(min_value, max_value)

    def set_const_weight(self, location, neuron_left, neuron_right, main_value):
        """
        location - <0/1> - 2 positions of edges(chose one):
            (input - hidden)-0 / (hidden - output)-1
        then set numbers of right and left neurons
        finally, set constant value
        """
        j1 = neuron_right - 1
        j2 = neuron_left - 1
        new = [j1, j2, main_value]
        if location == 0:
            self.list_const_values_0.append(new)
        elif location == 1:
            self.list_const_values_1.append(new)
        else:
            print("Values only in int range [0; 1]")

    def set_const_bias(self, neuron, main_value):
        """
        neuron - set number of neuron
            (1 - n)-input / (n+1 - n+k)-hidden / (n+k+1 - n+k+m)-output
        """
        new = [neuron, main_value]
        self.list_const_biases.append(new)
        print(self.list_const_biases)

    def set_learn_rate(self, value: float):
        self.learn_rate = value

    def set_epochs(self, value: float):
        self.epoch = value

    # ===============ACTIVATE FUNCTIONS================

    def set_activate_function(self, func):
        self.activation_f = func

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    def relu(self, x):
        return max(0, x)

    def deriv_sigmoid(self, x):
        # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def average_square(self, list_values: list):
        count = 0
        for i in range (len(list_values)):
            count += (list_values[i])**2
        count = count/len(list_values)
        return count

    def mse_loss(self, y_true, y_pred):
        # y_true and y_pred are numpy arrays of the same length.
        return ((y_true - y_pred) ** 2).mean()

    def mse_loss_nonp(self, y_true, y_pred):
        # [1, 2, 3], [1, 1, 1] renurn [0, 1, 2] --> [0, 1, 4] --> 5/3
        new_list = []
        for i in range (len(y_true)):
            new_list.append(y_true[i] - y_pred[i])
        return self.average_square(new_list)

    def mse_loss_square(self, y_true, y_pred):
        y_true_second = []
        y_pred_second = []
        for i in y_true:
            i = self.average_square(i)
            y_true_second.append(i)
        for i in y_pred:
            i = self.average_square(i)
            y_pred_second.append(i)

        y_true_second = np.array(y_true_second)
        y_pred_second = np.array(y_pred_second)
        return self.mse_loss(y_true_second, y_pred_second)

    def mse_loss_square_nonp(self, y_true, y_pred):
        y_true_second = []
        y_pred_second = []
        for i in y_true:
            i = self.average_square(i)
            y_true_second.append(i)
        for i in y_pred:
            i = self.average_square(i)
            y_pred_second.append(i)
        return self.mse_loss_nonp(y_true_second, y_pred_second)

    # ===================MAIN TRAIN==================

    def feedforward_output(self, x: list):
        h1 = 0
        hidden_neurons = [None] * self.deg_hidden_neurons
        ik = 0
        for i1 in self.list_inp_hid_weights:
            i0 = 0
            for i2 in i1:
                h1 += x[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons]
            hidden_neurons[ik] = h1
            ik += 1
            h1 = 0
        for j in range (self.deg_hidden_neurons):
            hidden_neurons[j] = self.activation_f(hidden_neurons[j])
        #hidden_neurons = [0.999999, 0.99999999, 0.99999]
        ik = 0
        h1 = 0
        output_neurons = [None] * self.deg_output_neurons
        for i1 in self.list_hid_out_weights:
            i0 = 0
            for i2 in i1:
                h1 += hidden_neurons[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons + self.deg_hidden_neurons]
            output_neurons[ik] = h1
            ik += 1
            h1 = 0
        for j in range (self.deg_output_neurons):
            output_neurons[j] = self.activation_f(output_neurons[j])
        return output_neurons

    def feedforward_hidden(self, x: list):
        h1 = 0
        hidden_neurons = [None] * self.deg_hidden_neurons
        ik = 0
        for i1 in self.list_inp_hid_weights:
            i0 = 0
            for i2 in i1:
                h1 += x[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons]
            hidden_neurons[ik] = int(h1)
            ik += 1
            h1 = 0
        for j in range(self.deg_hidden_neurons):
            hidden_neurons[j] = self.activation_f(hidden_neurons[j])
        return hidden_neurons

    def feedforward_output_sum(self, x: list):
        h1 = 0
        hidden_neurons = [None] * self.deg_hidden_neurons
        ik = 0
        for i1 in self.list_inp_hid_weights:
            i0 = 0
            for i2 in i1:
                h1 += x[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons]
            hidden_neurons[ik] = int(h1)
            ik += 1
            h1 = 0
        for j in range (self.deg_hidden_neurons):
            hidden_neurons[j] = self.activation_f(hidden_neurons[j])
        #hidden_neurons = [0.999999, 0.99999999, 0.99999]
        ik = 0
        h1 = 0
        output_neurons = [None] * self.deg_output_neurons
        for i1 in self.list_hid_out_weights:
            i0 = 0
            for i2 in i1:
                h1 += hidden_neurons[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons + self.deg_hidden_neurons]
            output_neurons[ik] = h1
            ik += 1
            h1 = 0
        return output_neurons

    def feedforward_hidden_sum(self, x: list):
        h1 = 0
        hidden_neurons = [None] * self.deg_hidden_neurons
        ik = 0
        for i1 in self.list_inp_hid_weights:
            i0 = 0
            for i2 in i1:
                h1 += x[i0] * i2
                i0 += 1
            h1 += self.list_biases[ik + self.deg_input_neurons]
            hidden_neurons[ik] = int(h1)
            ik += 1
            h1 = 0
        return hidden_neurons

    def train(self, data, all_y_trues):
        list_derivatives_0 = []
        list_derivatives_1 = []
        list_inp_hid_der = [[None]*self.deg_input_neurons]*self.deg_hidden_neurons
        list_hid_out_der = [[None]*self.deg_hidden_neurons]*self.deg_output_neurons

        number_one_neuron_output = 0
        number_i1 = 0
        number_i2 = 0
        number_i3 = 0
        number_i4 = 0
        for epoch in range(self.epochs):
            for x, y_trues in zip(data, all_y_trues):

                for one_neuron_output in y_trues:
                    # on iteration x - list trainers / one_neuron_output - y_true
                    y_pred = self.feedforward_output(x)
                    y_notall_pred = self.feedforward_hidden(x)
                    y_pred_sum = self.feedforward_output_sum(x)
                    y_notall_pred_sum = self.feedforward_hidden_sum(x)

                    p_L_p_ypred = -2 * (one_neuron_output - y_pred[number_one_neuron_output])

                    for i1 in self.list_inp_hid_weights:
                        for i2 in i1:
                            derivative_i2 = self.learn_rate
                            derivative_i2 *= p_L_p_ypred
                            derivative_i2 *= self.list_hid_out_weights[number_one_neuron_output][number_i1]
                            derivative_i2 *= self.deriv_sigmoid(y_pred_sum[number_one_neuron_output])
                            derivative_i2 *= x[number_i2]
                            derivative_i2 *= self.deriv_sigmoid(y_notall_pred_sum[number_i1])
                            list_derivatives_0.append(derivative_i2)
                            list_inp_hid_der[number_i1][number_i2] = i2 - derivative_i2
                            number_i2 += 1
                        number_i1 += 1
                        number_i2 = 0
                    number_i1 = 0

                    for i3 in self.list_hid_out_weights:
                        for i4 in i3:
                            derivative_i4 = self.learn_rate * p_L_p_ypred * y_notall_pred[number_i3] * self.deriv_sigmoid(y_pred_sum[number_i3])
                            list_derivatives_1.append(derivative_i4)
                            list_hid_out_der[number_i3][number_i4] = i4 - derivative_i4
                            number_i4 += 1
                        number_i4 = 0
                        number_i3 += 1
                    number_i3 = 0

                    for i1 in range (self.deg_hidden_neurons):
                        for i2 in range (self.deg_input_neurons):
                            self.list_inp_hid_weights[i1][i2] = list_inp_hid_der[i1][i2]

                    for i3 in range (self.deg_output_neurons):
                        for i4 in range (self.deg_hidden_neurons):
                            self.list_hid_out_weights[i3][i4] = list_hid_out_der[i3][i4]

                    list_inp_hid_der = [[None] * self.deg_input_neurons] * self.deg_hidden_neurons
                    list_hid_out_der = [[None] * self.deg_hidden_neurons] * self.deg_output_neurons
                    number_one_neuron_output += 1
                number_one_neuron_output = 0

            if epoch%10 == 0:
                massiv = [None]*len(data)
                for k in range(len(data)):
                    massiv[k] = self.feedforward_output(data[k])
                loss = self.mse_loss_square_nonp(all_y_trues, massiv)
                print(epoch, "=======================================", loss)
                self.x_means.append(epoch)
                self.y_means.append(loss)



#plt.plot(keki.x_means, keki.y_means)
#plt.show()

#5, 4 - ok, ok, ok, no, ok
#3, 5 - ok, ok, no, no, ok
#-4, 7 - no, no, no, no, no





