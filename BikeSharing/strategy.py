#coding:utf-8
import numpy as np

class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(
            loc=0.0, scale=self.hidden_nodes**-0.5, size=(hidden_nodes, input_nodes)
        )
        self.weights_hidden_to_output = np.random.normal(
            loc =0.0, scale=self.output_nodes**-0.5, size=(output_nodes, hidden_nodes)
        )

        self.lr = learning_rate

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
        self.activation_partial_function =lambda x : x * (1 - x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T


        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        output_errors_term = output_errors * self.activation_partial_function(final_outputs)

        hidden_errors = np.dot(output_errors_term, self.weights_hidden_to_output)
        hidden_errors_term = hidden_errors * self.activation_partial_function(hidden_outputs)[0]


        del_weights_input_to_hidden = np.dot(hidden_errors_term.T , inputs.T)
        del_weights_hidden_to_output = np.dot(output_errors_term.T , hidden_outputs.T)

        self.weights_input_to_hidden += self.lr * del_weights_input_to_hidden
        self.weights_hidden_to_output += self.lr * del_weights_hidden_to_output

    def run(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs =  np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

