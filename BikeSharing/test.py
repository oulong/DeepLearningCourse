import unittest
import numpy as np
import pandas as pd
from DataPreProcessing import data_path,rides
from strategy import NeuralNetwork


inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3],
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])




class TestMethods(unittest.TestCase):

    def test_data_path(self):
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_load(self):
        self.assertTrue(isinstance(rides, pd.DataFrame))

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        self.assertTrue(np.all(network.activation_function(0.5) == 1 /(1 + np.exp(-0.5))))

    def test_train(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)

        print(network.weights_hidden_to_output)
        print(network.weights_input_to_hidden)

        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))
    def test_run(self):
        # Test correctness of run method
        '''
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))
        '''
        pass

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)