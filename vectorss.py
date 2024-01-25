# import numpy as np
#
# input = [2.1, 2.2, -1.2]
# weight = [5, 3.1, 2]
# bias = 3;
# output = input[0] * weight[0] * input[1] * weight[1] * input[2] * weight[2] + bias
# print(output)
# input = [2.1, 2.2, -1.2]
# weight1 = [4, 5, 2, ]
# weight2 = [2.8, -1.6, 5.8]
# weight3 = [2.1, 5.1, 3, .2]
# bias1 = 2
# bias2 = 3
# bias3 = 2.7
# output1 = input[0] * weight1[0] * input[1] * weight2[1] * input[2] * weight3[2] + bias
# print(output1)
# array = [2, 2.1, 1, 2.2]
# weight = np.array([[4, 5, -2], [2.8, -16, 5.8], [2.5, 16, 3.2]])
# bias = np.array([2, 3, 2.7])
# result = np.dot(input, weight) + bias
# print('---------------------------------------------')
# print(result)
# ger_out = []
# for newron_weight, newron_bias in(weight, bias):
#     newron_output = 0
# for n_weight, n_input, in (newron_weight, input):
#     newron_output += n_weight * n_input
#     newron_output += newron_bias
# ger_out.append(newron_output)
# print(ger_out)
import numpy as np
from matplotlib import pyplot as plt

x = np.array([2.1, 2.2, -1.8])
x = np.random.randn(3, 4)
input = [3.4, 2.3, 3.4]
bias = [3.4, 4.5]
weight = [[5, 3.1], [2.3, -2.8], [3.1, 2.5]]
weight = np.array(weight)
print(weight.shape)
print(weight.ndim)
# print(type(x))
# print(x.shape)
# print('Number of dimension for x', x.ndim)
# print(x)
y = np.dot(weight.T, input) + bias
print(y)
print('The transpose is:\n', weight.T)


class Dense_layer:
    def __init__(self, inputs, neurons):
        self.weight = 0.2 * np.random.randn(inputs, neurons)
        self.bias = 0.3 * np.random.randn(1, neurons)

    def forward(self, inputdate):
        self.output = np.dot(inputdate, self.weight) * self.bias
        return self.output


layer1 = Dense_layer(4, 5)
layer1.forward(x)
layer2 = Dense_layer(5, 2)
layer2.forward(layer1.output)
print('The dense_layer_outp:\n', layer1.output)
print('ANN_output:\n', layer2.output)
X = np.load("C:/Users/user/Downloads/X.npy")
# print('The signs digt data set:\n', X)
print('X250:\n', X[250])
plt.imshow(X[250])
'''plt.show()
print('X250 shape:\n', X[250].shape)
print('X250 dimension:\n', X[250].ndim)'''


class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


Activation1 = Activation_Relu()
Activation1.forward(layer2.output)
print('Output of Activation_Relu:\n', Activation1.output)


class stepup_acivation:
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 0)


Activation2 = stepup_acivation()
Activation2.forward(layer2.output)
print('Output of stepup_activation is:\n', Activation2.output)


class softmax_activation:
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        exp_values_aftermv = exp_values - np.max(exp_values)
        self.output_flormvd = exp_values_aftermv / np.sum(exp_values_aftermv)
        exp_values_total = np.sum(exp_values)
        # self.probabilities =exp_values / exp_values_total
        probabilities = exp_values / exp_values_total
        probabilities = exp_values_aftermv / np.sum(exp_values_aftermv)
        # return probabilities
        # self.output = probabilities
Activation3 = softmax_activation()
Activation3.forward(layer2.output)
# print('Output of softmax_activation is:\n', Activation3.output)
print('Overflow:\n', Activation3.output_flormvd)