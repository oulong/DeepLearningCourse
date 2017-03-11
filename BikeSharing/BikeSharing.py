#coding:utf-8

import sys
import numpy as np
from DataPreProcessing import (
    test_features,test_targets
    , train_features,train_targets
    , val_features, val_targets
    , scaled_features)
from strategy import NeuralNetwork

epochs = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)


def MSE(y, Y):
    return np.mean((y-Y)**2)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    batch = np.random.choice(train_features.index, size=128)
    for record,target in zip(train_features.ix[batch].values
                            , train_targets.ix[batch]['cnt']):
        network.train(record, target)

    train_loss = MSE(network.run(train_features), train_targets)
    val_loss = MSE(network.run(val_features), val_targets)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

fig, ax = plt.subplots(figsize=(8,4))
mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)