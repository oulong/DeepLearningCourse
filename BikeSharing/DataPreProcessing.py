#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv

	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit :
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
'''
data_path = "Bike-Sharing-Dataset/hour.csv"
rides = pd.read_csv(data_path)
#print(len(rides.index))
# rides[:24*10].plot(x='dteday', y='cnt')
# plt.show()
dumy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dumy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday'] + dumy_fields
data = rides.drop(fields_to_drop, axis = 1)

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for field in quant_features:
    mean, std = data[field].mean(), data[field].std()
    scaled_features[field] = [mean, std]
    data.loc[:, field] = (data[field] - mean) / std

test_data = data[-21*24:]

data = data[:-21*24]

target_field = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_field, axis=1), data[target_field]
test_features, test_targets = test_data.drop(target_field, axis = 1), test_data[target_field]

train_features, train_targets = features[:-60*24],targets[:-60*24]
val_features, val_targets = features[-60*24:],targets[-60*24:]

