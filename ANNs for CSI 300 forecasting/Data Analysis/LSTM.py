import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import datetime
import os 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#exponential moving average for 10
def eMovingAverage(x,n):
	mv10 = []
	multiplier = 2/(n+1)

	mv10.append(np.sum(x[0:n]) / n)

	for i in range(n,len(x)):
		mv10.append(((x[i] - mv10[i-(n)]) * multiplier) + mv10[i-(n)])
	return mv10

## simple moving average 
def movingAverage(x,n):
	mv = []
	for i in range(n-1,len(x)):
		mv.append(np.sum(x[i-n-1:i])/n)
	return mv

## ADL 
def adl_cal(high,low,close,volume):

	acumulator = np.zeros(len(low))
	multiplier1 = ((close[0] - low[0]) - (high[0] - close[0])) / (high[0] - low[0])
	acumulator[0] = multiplier1 * volume[0]

	for i in range(1,len(low)):
		flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
		flow_volume = flow_multiplier * volume[i]
		acumulator[i] = acumulator[i-1] + flow_volume
	return	acumulator

## momentum ROC
def momentum(x):
	acumulator = []
	counter = 0
	for i in range(10,len(x)):
		acumulator.append(((x[i] - x[counter])/x[i]) * 100)
		counter = counter + 1
	return	acumulator

## len check
def lenCheck(x,y):
	return (len(x) == len(y))

## stochastic oscillator including the same day --- 10 days
def stochasticOscillator(x):
	acumulator = []
	for i in range(9,len(x)):
		recent_close = x[i]
		high = np.max(x[i-9:i])
		low = np.min(x[i-9:i])
		acumulator.append(((recent_close - low) / (high - low)) * 100)
	return acumulator

def profit(x):
	acumulator = []
	for i in range(len(x)-1):
		result = x[i] - x[i+1]
		if result > 0:
			acumulator.append(1)
		else:
			acumulator.append(0) 
	return acumulator

## RSI
def rsiCalculation(x):
	loss = []
	gain = []
	rsi = []
	for i in range(1,11):
		result = x[i] - x[i-1]

		if result >= 0.0:
			gain.append(result)
		else:
			loss.append(abs(result))

	first_gains = np.sum(gain) / 10
	first_loss = np.sum(loss) / 10

	

	rs = first_gains/ first_loss
	rsi.append(100 - (100 / (1+rs)))

	average_gain = []
	average_loss = []

	average_gain.append(first_gains)
	average_loss.append(first_loss)

	for i in range(11,len(x)):

		result = x[i] - x[i-1]
		averageGain = 0
		averageLoss = 0

		if result >= 0.0:
			averageGain = ((average_gain[i-11] * 9) + result) / 10
			averageLoss = ((average_loss[i-11] * 9)) / 10

			average_gain.append(averageGain)
			average_loss.append(averageLoss)
		else:
			averageGain = ((average_gain[i-11] * 9)) / 10
			averageLoss = ((average_loss[i-11] * 9) + abs(result)) / 10

			average_gain.append(averageGain)
			average_loss.append(averageLoss)

		rs = averageGain/averageLoss
		rsi.append(100 - (100 / (1+rs)))

	return rsi

##data normalization
def normalization(x):
	mean = np.mean(x)
	deviation = np.std(x)
	norm_data = (x - mean) / deviation
	return mean,deviation,norm_data

def gaf_Image(x):
	data = []
	for i in range(len(x)):
		arccos = np.arccos(x[i])
		matrix = arccos.T + arccos
		data.append(np.cos(matrix))

	return np.array(data)

def data_to_image(x,days):
	conv_data = []
	for z in range(0,len(x)-days):
		maximum = np.amax(x[z:z+days])
		minimum = np.amin(x[z:z+days])
		numerator = ((x[z:z+days] - maximum) + (x[z:z+days] - minimum))
		norm = numerator / (maximum - minimum)
		arccos = np.arccos(norm).reshape(1,days_taken)
		matrix = arccos.T + arccos
		conv_data.append(matrix)
	return np.reshape(np.array(conv_data),[-1,days,days])

def image_ordering(image):
	number = len(image[0])
	x = len(image[0][0])
	y = len(image[0][0][0])
	z = len(image)

	acc = np.zeros([number,x,y,z])	

	for i in range(z):
		for j in range(number):
			for q in range(x):
				for g in range(y):
					acc[j][q][g][i] = image[i][j][q][g]
	return acc		

##
def max_min(x):
	print(np.min(x),np.max(x))

def prediction_normalization(x):
	maximum = np.amax(x)
	minimum = np.amin(x)
	numerator = ((x - maximum) + (x - minimum))
	norm = numerator / (maximum - minimum)
	return norm, numerator, (maximum - minimum)

#getting CSI data
data = pd.read_csv("CSIData2.csv")

date = np.array(data['Name'])
csi_high = np.array(data['PRICE HIGH'],dtype = float)
csi_low = np.array(data['PRICE LOW'],dtype = float)
csi_close = np.array(data['PRICE INDEX'],dtype = float)
csi_volume = np.array(data['VOLUME'],dtype = float)

logic = np.isnan(csi_volume)


#for removing nans
date = date[~logic]
csi_high = csi_high[~logic]
csi_low = csi_low[~logic]
csi_close = csi_close[~logic]
csi_volume = csi_volume[~logic]

dates = [datetime.datetime.strptime(date1, '%m/%d/%Y') for date1 in date]

#exchange data
data2 = np.genfromtxt("exchange.csv",delimiter = ",",skip_header=2)
exchange = data2[:,1]
exchange = exchange[~logic]

##features to use exponential moving average 10 days and 30 days, momentum for 10 days, forex exchange rate too, ADL, 3 days close price, Stochastic oscillator

# 10 days exponential moving average
emove10_close =  eMovingAverage(csi_close,10)

## 10 days momentum
momentum10_close = momentum(csi_close)

# 10 days stochastic oscillator && 3 period moving average of oscillator 
oscillator10_close = stochasticOscillator(csi_close)
oscilaltor10movingAverage_close = movingAverage(oscillator10_close,3)
## 30 days exponential moving average
emove30_close = eMovingAverage(csi_close,30)

## RSI calculation 10 days
rsi_close = rsiCalculation(csi_close)


## Adjusting data length for training
adjustment = len(dates) - len(emove30_close)
csi_high = csi_high[adjustment:]
csi_low = csi_low[adjustment:]
csi_close = csi_close[adjustment:]
csi_volume = csi_volume[adjustment:]
dates = dates[adjustment:]

momentum10_close = momentum10_close[adjustment-10:-1]
emove10_close = emove10_close[adjustment-9:-1]
oscillator10_close = oscillator10_close[adjustment-9:-1]
oscilaltor10movingAverage_close = oscilaltor10movingAverage_close[adjustment-11:-1]
rsi_close = rsi_close[adjustment-10:-1]
emove30_close = emove30_close[0:-1]

## Accumulation distribution line ADL
adl = adl_cal(csi_high,csi_low,csi_close,csi_volume)
adl = adl[0:-1]

sample = len(momentum10_close)
sample_8 = math.ceil(sample *.8)
sample_2 = math.floor(sample * .2)


norm_momentum10_close = normalization(momentum10_close[:sample_8])
norm_emove10_close = normalization(emove10_close[:sample_8])
norm_emove30_close = normalization(emove30_close[:sample_8])
norm_oscillator10_close = normalization(oscillator10_close[:sample_8])
norm_oscilaltor10movingAverage_close = normalization(oscilaltor10movingAverage_close[:sample_8])
norm_rsi_close = normalization(rsi_close[:sample_8])
norm_adl = normalization(adl[:sample_8])
norm_csi_close = normalization(csi_close[:sample_8])


data_set = np.vstack((norm_momentum10_close[2],norm_emove10_close[2],norm_emove30_close[2],norm_oscillator10_close[2],norm_oscilaltor10movingAverage_close[2],norm_rsi_close[2],norm_adl[2]))
data_set = data_set.T


norm_momentum10_close2 = (momentum10_close[sample_8:sample_8+sample_2] - norm_momentum10_close[0]) / norm_momentum10_close[1]
norm_emove10_close2 =  (emove10_close[sample_8: sample_8+sample_2] - norm_emove10_close[0])  / norm_emove10_close[1] 
norm_emove30_close2 =  (emove30_close[sample_8: sample_8+sample_2] - norm_emove30_close[0])  / norm_emove30_close[1]
norm_oscillator10_close2 =  (oscillator10_close[sample_8:sample_8+sample_2] - norm_oscillator10_close[0])  / norm_oscillator10_close[1]
norm_oscilaltor10movingAverage_close2 =  (oscilaltor10movingAverage_close[sample_8:sample_8+sample_2] - norm_oscilaltor10movingAverage_close[0])  / norm_oscilaltor10movingAverage_close[1]
norm_rsi_close2 =  (rsi_close[sample_8:sample_8+sample_2] - norm_rsi_close[0])  / norm_rsi_close[1]
norm_adl2 = (adl[sample_8:sample_8+sample_2] - norm_adl[0])  / norm_adl[1]
norm_csi_close2 =  (csi_close[sample_8:sample_8+sample_2] - norm_csi_close[0]) / norm_csi_close[1]

training_set = np.vstack((norm_momentum10_close2,norm_emove10_close2,norm_emove30_close2,norm_oscillator10_close2,norm_oscilaltor10movingAverage_close2,norm_rsi_close2,norm_adl2))
training_set = training_set.T

element_size = 7
time_steps = 1
hidden_neurons = 8
batch_size = 32
learning_rate = 0.01
epochs = 1500
sequence_array = np.ones(batch_size) * time_steps

initial = tf.keras.initializers.he_normal()
input_x = tf.placeholder(shape = [None,time_steps,element_size],dtype = tf.float32)
y_true = tf.placeholder(shape = (None),dtype = tf.float32)
w1 = tf.Variable(initial([hidden_neurons,1]))
b1 = tf.Variable(initial([1]))

lstm_cell = tf.contrib.rnn.LSTMCell(hidden_neurons,initializer = initial,forget_bias = 1.0)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_x, dtype=tf.float32)

final_output = tf.matmul(states[1],w1) + b1
cost_function =  tf.reduce_mean(tf.square(y_true - final_output)) 
accuracy = tf.reduce_mean(tf.square(y_true - final_output))

optimization = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimization.minimize(cost_function)


y_test = []
y_train = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for steps in range(epochs):	
		for i in range(0,sample_8,batch_size):			
			sess.run(train,feed_dict = {input_x:data_set[i: i + batch_size].reshape(-1,time_steps,7),y_true:csi_close[i+1: i + batch_size + 1]})
	for i in range(sample_8):
		ans = sess.run(final_output,feed_dict = {input_x:data_set[i].reshape(-1,time_steps,7),y_true:csi_close[i + 1]})
		y_train.append(ans[0][0])
	for i in range(sample_2):
		ans1 = sess.run(final_output,feed_dict = {input_x:training_set[i].reshape(-1,time_steps,7),y_true:csi_close[sample_8 + i+1]})
		y_test.append(ans1[0][0])

y_test= np.array(y_test)
np.savetxt('LSTMpredict.csv',y_test,delimiter = ',')

mse = np.sum(np.square(csi_close[1:sample_8+1] - y_train))/len(y_train) 
print('MSE training',mse)
print('MAPE training', np.mean(abs((csi_close[1:sample_8+1] - y_train))/ csi_close[1:sample_8+1]))

print('MSE testing',np.mean(np.square(csi_close[sample_8 + 1: sample_2 + 1 + sample_8 ] - y_test)))
print('MAPE testing', np.mean(abs((csi_close[sample_8 + 1: sample_2 + 1 + sample_8 ] - y_test))/ csi_close[sample_8 + 1: 1 + sample_8 + sample_2]))
