import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
import random
import timeit

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		print("a:" + str(a))
		dataX.append(a)
		dataY.append(dataset[i + look_back], 0)
		print("dataX: " + str(dataX))
		print("dataY: " + str(dataY))


	return pd.DataFrame(dataX), pd.DataFrame(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset

# dataframe = pd.read_csv("DATA.csv")# delimiter=",")

# dataframe = read_csv('TEST.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataframe["Column"]= dataframe["Day1"].apply(numpy.fromstring, sep=",")
# print("Data: " + str(dataframe))

dataframe = pd.DataFrame([[[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [1]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [0]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [1]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [1]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [0]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649], [0]]])
#dataset = dataframe.values


# rows, columns = dataframe.shape

# for r in range(rows):
# 	for c in range(columns):
# 		TEST[r][c] = [numpy.random.randint(0, 1024),numpy.random.randint(0, 1024),
# 		numpy.random.randint(0, 1024), numpy.random.randint(0, 1024),
# 		numpy.random.randint(0, 1024),numpy.random.randint(0, 1024), numpy.random.randint(0, 1024),
# 		numpy.random.randint(0, 1024), numpy.random.randint(0, 1024), numpy.random.randint(0, 1024)]


# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# train = dataset[0:train_size,:]
# reshape into X=t and Y=t+1
look_back = 1
#trainX = dataset
#trainX, trainY = create_dataset(dataframe, look_back)



# trainX = dataset[[[3]]][:11]
# trainY = dataset[[[3]]][10]

trainX = numpy.array([[[random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)]],
			  [[890, 123, 496, 987, 231, 103, 921, 5, 932, 1],
			  [655, 234, 477, 945, 267, 178, 872, 8, 871, 28],
			  [532, 328, 430, 911, 291, 239, 837, 297, 834, 100],
			  [419, 345, 371, 854, 347, 241, 719, 472, 719, 146],
			  [387, 401, 364, 813, 380, 302, 701, 563, 704, 238],
			  [356, 457, 326, 768, 418, 386, 680, 897, 658, 293],
			  [324, 489, 281, 732, 518, 428, 618, 912, 613, 327],
			  [276, 502, 234, 697, 892, 460, 537, 953, 547, 369],
			  [239, 534, 211, 623, 923, 512, 511, 984, 513, 428],
			  [200, 589, 187, 510, 1001, 526, 489, 999, 508, 450],
			  [107, 733, 162, 476, 1020, 574, 427, 1007, 473, 523]],
			  [[random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)]],
			  [[890, 100, 1011, 400, 102, 33, 831, 2, 888, 69],
			  [755, 150, 932, 378, 123, 67, 810, 4, 823, 137],
			  [707, 189, 874, 325, 223, 89, 702, 6, 786, 182],
			  [627, 203, 810, 317, 289, 103, 628, 12, 752, 213],
			  [612, 254, 764, 286, 368, 154, 614, 24, 714, 257],
			  [581, 278, 731, 265, 483, 187, 530, 378, 638, 270],
			  [532, 327, 709, 231, 519, 260, 519, 463, 615, 307],
			  [470, 345, 687, 210, 571, 293, 473, 548, 579, 359],
			  [463, 427, 496, 156, 610, 317, 436, 794, 543, 372],
			  [416, 461, 358, 132, 629, 372, 411, 827, 412, 403],
			  [337, 790, 219, 129, 647, 480, 376, 898, 378, 485]],
			  [[890, 341, 785, 823, 14, 389, 897, 76, 893, 115],
			  [654, 489, 721, 811, 313, 413, 723, 102, 832, 221],
			  [610, 523, 689, 789, 378, 451, 638, 128, 767, 247],
			  [592, 595, 624, 740, 416, 499, 617, 156, 743, 286],
			  [573, 604, 601, 634, 485, 529, 537, 182, 721, 326],
			  [541, 657, 598, 620, 529, 530, 528, 287, 678, 408],
			  [494, 692, 528, 618, 545, 672, 511, 302, 654, 473],
			  [476, 703, 428, 520, 632, 690, 409, 683, 632, 562],
			  [399, 754, 403, 506, 677, 712, 312, 792, 229, 681],
			  [347, 892, 376, 423, 692, 728, 300, 900, 217, 703],
			  [292, 1021, 315, 398, 704, 739, 242, 1015, 148, 756]],
			  [[random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)]],
			  [[910, 153, 834, 993, 0, 388, 1011, 76, 754, 120],
			  [893, 187, 811, 912, 178, 414, 819, 103, 743, 218],
			  [827, 192, 713, 826, 234, 467, 726, 133, 722, 257],
			  [753, 293, 700, 808, 542, 499, 723, 156, 698, 286],
			  [712, 304, 628, 762, 679, 529, 680, 204, 684, 333],
			  [692, 317, 615, 739, 701, 530, 654, 287, 642, 417],
			  [624, 339, 598, 659, 739, 678, 523, 356, 521, 476],
			  [613, 376, 573, 613, 744, 703, 511, 789, 509, 561],
			  [572, 405, 510, 600, 790, 712, 499, 792, 428, 688],
			  [530, 627, 428, 524, 812, 764, 483, 902, 411, 704],
			  [516, 713, 408, 510, 858, 819, 412, 1020, 338, 787]],
			  [[random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)]],
			  [[914, 230, 854, 994, 130, 317, 1011, 18, 755, 84],
			  [899, 239, 821, 923, 178, 400, 907, 79, 740, 139],
			  [832, 246, 769, 845, 238, 415, 827, 108, 721, 212],
			  [765, 289, 723, 808, 425, 483, 723, 123, 697, 283],
			  [713, 301, 711, 761, 613, 512, 689, 196, 684, 338],
			  [641, 369, 621, 744, 701, 532, 643, 203, 643, 479],
			  [628, 427, 606, 659, 749, 572, 527, 277, 522, 483],
			  [619, 521, 573, 617, 786, 698, 510, 309, 506, 513],
			  [523, 589, 518, 600, 791, 712, 499, 462, 429, 627],
			  [513, 627, 427, 525, 814, 764, 477, 523, 411, 704],
			  [512, 718, 410, 512, 857, 1002, 321, 600, 337, 832]],
			  [[random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)],
			  [random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023), random.randint(0, 1023)]]])
# trainX = dataframe.drop[:,:,10]
# trainX = dataframe.iloc[:,:, 0:11].values

trainY = pd.DataFrame([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])



# trainX = dataframe[][][:11]
# trainY = dataframe[][][10]

print("trainX", trainX.shape)
print("trainX stuff", trainX)

# trainX = numpy.reshape(trainX, trainX.shape + (1,))

# testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
print(trainX.shape)
model = Sequential()
model.add(LSTM(40, input_shape=(11,10),return_sequences=True))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
print(trainX.shape)
print(trainY.shape)

 #    keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid',
	# use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
	# bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
	# bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
	# bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False,
	# return_state=False, go_backwards=False, stateful=False, unroll=False)
 #    model = Sequential()
 #    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
 #    model.add(LSTM(hidden_size, return_sequences=True))
 #    model.add(LSTM(hidden_size, return_sequences=True))
 #    if use_dropout:
 #    	model.add(Dropout(0.5))
 #    model.add(TimeDistributed(Dense(vocabulary)))
 #    model.add(Activation('softmax'))

#input_shape=X_train.shape[1:]
#TRAINY = TARGETS
model.fit(trainX, trainY, epochs=100, batch_size=150, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions

# print("Final Accuracy: ")
# print(cross_val_score(my_classifier, DATA_INPUTS, DATA_TARGETS, cv=3, scoring="accuracy").mean())


# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()