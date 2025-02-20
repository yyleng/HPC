import torch
import numpy as np
bikes_numpy = np.loadtxt(
    "./data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
bikes 

# the dataset has been cleaned, that's mean we have no noise in the data
# now we change the data shape to NxCxL(batch, cols, hours)
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape
daily_bikes.stride()

# now, we focus on the first day and change the wweather column to one-hot encoding
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:,9]
weather_onehot.scatter_(
dim=1,
 index=first_day[:,9].unsqueeze(1).long() - 1,
 value=1.0)
# connect the weather one-hot encoding to the first day
torch.cat((bikes[:24], weather_onehot), 1)[:1]
# now we do the same for the whole dataset
daily_weather_onehot = torch.zeros(daily_bikes.shape[0],4,daily_bikes.shape[2])
daily_weather_onehot.shape
daily_weather_onehot.scatter_(1,daily_bikes[:,9,:].long().unsqueeze(1)-1,1.0)
daily_weather_onehot.shape
# then connect the C dims
torch.cat((daily_bikes, daily_weather_onehot), 1)
# 将变量重新调整到[0.0,1.0]或[−1.0,1.0]是我们对所有变量都要做的事情，如温度（数据集中
# 的第 10 列）。这对训练过程是有益的
temp = daily_bikes[:, 10, :]
temp.shape
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:,10,:] = (daily_bikes[:,10,:] - temp_min)/(temp_max-temp_min)
# we also can mean and std
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp))
 / torch.std(temp))
