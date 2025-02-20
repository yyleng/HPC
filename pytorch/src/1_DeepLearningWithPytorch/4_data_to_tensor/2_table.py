import numpy
import csv
import torch

#--------------------------------
# use numpy load csv file
wine_path = "./data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy = numpy.loadtxt(wine_path, dtype=numpy.float32, delimiter=";", skiprows=1)
wine_numpy.shape
wine_numpy

winet = torch.from_numpy(wine_numpy)
winet.shape
winet

#--------------------------------
# the above the last col is the target, so we need to split the data to feature and target
data = winet[:, :-1]
target = winet[:, -1]
data.shape
data
target.shape
target
## if the target is a string, we can select an integer to represent it

#--------------------------------
# one-hot encoding
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.long().unsqueeze(1), 1.0)
target_onehot
## feature: unordered categorical data, such as color, we can use one-hot encoding
## scenario: text classification, we can use one-hot encoding to represent the word
## scenario: image classification, we can use one-hot encoding to represent the category
## scenario: advice system, we can use one-hot encoding to represent the user's behavior

#--------------------------------
# normalize data
data_mean = torch.mean(data, dim=0)
data_mean
data_std = torch.std(data, dim=0)
data_std
data_normalized = (data - data_mean) / data_std
data_normalized

#--------------------------------
# now we split the data. < 3 is bad, >= 7 is good
bad_indexes = target <= 3
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()
bad_data = data[bad_indexes] # Advanced index
bad_data.shape

bad_data = data[target <= 3] # bad wine
mid_data = data[(target > 3) & (target < 7)] # medium wine, use & to combine two conditions
good_data = data[target >= 7] # good wine

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
col_list = next(csv.reader(open(wine_path), delimiter=';'))
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum() 
