#coding:utf-8
import numpy as np
import time
import sys
from collections import Counter

class SentimentNetwork:

    def __init__(self, reviews, labels, polarity_cutoff = 0.1, min_count = 10, hidden_node = 10, learning_rate = 0.1):

        np.random.seed(1)

        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)

        self.init_network(self.review_vocab_size,  hidden_node, 1, learning_rate)


    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):

        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()
        # frequency_frequency = Counter()

        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        #for word, cnt in total_counts.most_common():
        #    frequency_frequency[cnt] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt > 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))

        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):

                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if abs(pos_neg_ratios[word]) > polarity_cutoff:
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)


        self.review_vocab = list(review_vocab)


        label_vocab = set()
        for label in labels:
            label_vocab.add(label.upper())

        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i



    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weight_i_h = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weight_h_o = np.random.normal(0.0, self.output_nodes**-0.5, size=(self.hidden_nodes, self.output_nodes))

        self.input_layer = np.zeros((1, input_nodes))
        self.hidden_layer = np.zeros((1, hidden_nodes))


    def update_input_layer(self, review):

        self.input_layer *= 0
        for word in review.split(" "):
            if word in self.word2index.keys():
                self.input_layer[0][self.word2index[word]] = 1

    def get_target_for_label(self, label):
        if (label == 'POSITIVE'):
            return 1
        else:
            return 0

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmod_output_2_derivative(self, x):
        return x * (1 - x)

    def train(self, training_reviews_raw, training_labels):

        # new add
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(' '):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_reviews.append(indices)


        assert(len(training_reviews) == len(training_labels))

        corrent_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]

            #input layer
            #self.update_input_layer(review)


            #hidden_layer = self.input_layer.dot(self.weight_i_h)
            #==> to modify
            self.hidden_layer *= 0
            for index in review:
                self.hidden_layer += self.weight_i_h[index]

            out_layer = self.sigmod(self.hidden_layer.dot(self.weight_h_o))


            out_layer_error = out_layer - self.get_target_for_label(label)
            out_layer_delta = out_layer_error * self.sigmod_output_2_derivative(out_layer)

            hidden_layer_error = out_layer_delta.dot(self.weight_h_o.T)
            hidden_layer_delta = hidden_layer_error


            self.weight_h_o -= self.hidden_layer.T.dot(out_layer_delta) * self.learning_rate

            #self.weight_i_h -= self.input_layer.T.dot(hidden_layer_delta) * self.learning_rate
            #=> to modify
            for index in review:
                self.weight_i_h[index] -= hidden_layer_delta[0] * self.learning_rate

            if np.abs(out_layer_error) < 0.5:
                corrent_so_far +=1


            review_per_second = i / float(time.time() - start)

            sys.stdout.write("\rTrain Progress:" + str(100*i/(len(training_reviews)))[0:4]
                             + "% Speed(revies/sec):" + str(review_per_second)[0:5]
                             + "% #Corrent: " + str(corrent_so_far)
                             + "  #Training Accuracy: " + str(corrent_so_far * 100 / float(i + 1))[0:4] + "%")
            if i > 0 and i % 2500 == 0:
                print("")

    def run(self, review):

        # self.update_input_layer(review.lower())

        # hidden_layer = self.input_layer.dot(self.weight_i_h)
        self.hidden_layer *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.hidden_layer += self.weight_i_h[index]

        out_layer = self.sigmod(self.hidden_layer.dot(self.weight_h_o))

        if  out_layer[0] > 0.5:
            return 'POSITIVE'
        else:
            return "NEGATIVE"


    def test(self, test_reviews, test_labels):

        assert(len(test_reviews) == len(test_labels))

        corrent = 0

        start = time.time()

        for i in range(len(test_reviews)):

            label = self.run(test_reviews[i])
            if label == test_labels[i]:
                corrent += 1

            review_per_second = i / ((float(time.time() - start)) + 0.01)

            sys.stdout.write("\rTest Progress:" + str(100*i/(len(test_reviews)))[0:4]
                             + "% Speed(revies/sec):" + str(review_per_second)[0:5]
                             + "% #Corrent: " + str(corrent)
                             + " #Training Accuracy: " + str(corrent * 100 / float(i + 1))[0:4] + "%")



