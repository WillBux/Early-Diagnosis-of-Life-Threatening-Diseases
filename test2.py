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

trainX = numpy.array([[[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]],
			  [[890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [654, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649],
			  [890, 123, 496, 987, 231, 903, 862, 214, 367, 649]]])
# trainX = dataframe.drop[:,:,10]
# trainX = dataframe.iloc[:,:, 0:11].values

trainY = pd.DataFrame([1, 0, 1, 1, 0, 0])



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
model.add(LSTM(4, input_shape=(11,10),return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
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
model.fit(trainX, trainY, epochs=10, batch_size=15, verbose=2)
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