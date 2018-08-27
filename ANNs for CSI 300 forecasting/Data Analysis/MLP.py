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
##
def max_min(x):
	print(np.min(x),np.max(x))

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

momentum10_close = momentum10_close[adjustment-10:-1]
emove10_close = emove10_close[adjustment-9:-1]
oscillator10_close = oscillator10_close[adjustment-9:-1]
oscilaltor10movingAverage_close = oscilaltor10movingAverage_close[adjustment-11:-1]
rsi_close = rsi_close[adjustment-10:-1]
emove30_close = emove30_close[0:-1]

## Accumulation distribution line ADL
adl = adl_cal(csi_high,csi_low,csi_close,csi_volume)
adl = adl[0:-1]

#print(len(momentum10_close) , len(emove10_close) , len(oscillator10_close) , len(oscilaltor10movingAverage_close) , len(rsi_close) , len(emove30_close) , len(adl),len(dates))

## data normalization
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
norm_exchange = normalization(exchange[:sample_8])

data_set = np.vstack((norm_momentum10_close[2],norm_emove10_close[2],norm_emove30_close[2],norm_oscillator10_close[2],norm_oscilaltor10movingAverage_close[2],norm_rsi_close[2],norm_adl[2]))
data_set = data_set.T

norm_momentum10_close1 = (momentum10_close[sample_8:sample] - norm_momentum10_close[0]) / norm_momentum10_close[1]
norm_emove10_close1 =  (emove10_close[sample_8:sample] - norm_emove10_close[0])  / norm_emove10_close[1] 
norm_emove30_close1 =  (emove30_close[sample_8:sample] - norm_emove30_close[0])  / norm_emove30_close[1]
norm_oscillator10_close1 =  (oscillator10_close[sample_8:sample] - norm_oscillator10_close[0])  / norm_oscillator10_close[1]
norm_oscilaltor10movingAverage_close1 =  (oscilaltor10movingAverage_close[sample_8:sample] - norm_oscilaltor10movingAverage_close[0])  / norm_oscilaltor10movingAverage_close[1]
norm_rsi_close1 =  (rsi_close[sample_8:sample] - norm_rsi_close[0])  / norm_rsi_close[1]
norm_adl1 = (adl[sample_8:sample] - norm_adl[0])  / norm_adl[1]
norm_csi_close1 =  (csi_close[sample_8:sample] - norm_csi_close[0]) / norm_csi_close[1]
norm_exchange1 =  (exchange[sample_8:sample] - norm_exchange[0])  / norm_exchange[1]

training_set = np.vstack((norm_momentum10_close1,norm_emove10_close1,norm_emove30_close1,norm_oscillator10_close1,norm_oscilaltor10movingAverage_close1,norm_rsi_close1,norm_adl1))
training_set = training_set.T

num_input_neurons = 7
num_hidden_neurons_1 = 10
num_output_neurons = 1 
learning_rate = 0.01
lambda1 = 0.01
epochs = 2000

input_x = tf.placeholder(dtype = tf.float32,shape = (None,num_input_neurons))
y_true = tf.placeholder(dtype = tf.float32,shape = (None))


weight_1 = tf.get_variable(name = "w1",shape=(num_input_neurons, num_hidden_neurons_1), initializer=tf.keras.initializers.he_normal())
b_1 = tf.get_variable(name = "b1",shape=(1, num_hidden_neurons_1), initializer=tf.contrib.layers.xavier_initializer())
weight_2 = tf.get_variable(name = "w2",shape=(num_hidden_neurons_1,num_output_neurons), initializer=tf.keras.initializers.he_normal())
b_2 = tf.get_variable(name = "b2",shape=(1, num_output_neurons), initializer=tf.contrib.layers.xavier_initializer())

## feed forward 
z_1 = tf.add(tf.matmul(input_x,weight_1),b_1)
a_1 = tf.tanh(z_1)
z_2 = tf.add(tf.matmul(a_1,weight_2),b_2)


weight_1_square = tf.square(weight_1)
weight_2_square = tf.square(weight_2)

weight_1_sum = tf.reduce_sum(weight_1_square)
weight_2_sum = tf.reduce_sum(weight_2_square)
weight_sum = weight_1_sum + weight_2_sum	

cost_loss = tf.reduce_mean(tf.square(y_true - z_2)) 
accuracy = tf.reduce_mean(tf.square(y_true - z_2)) 
optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate, beta1 = 0.9,beta2=0.999, epsilon=1e-08,)
train = optimizer.minimize(cost_loss)



batch_size = 32

y_predicted = []
y_predicted1 = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	# training process
	for steps in range(0,epochs):
		for i in range(0,sample_8,batch_size):			
			sess.run(train,feed_dict = {input_x:data_set[i: i + batch_size],y_true:csi_close[i+1: i + batch_size + 1]})
	#checking training results
	for i in range(sample_8):
		ans = sess.run(z_2,{input_x:data_set[i].reshape(1,7),y_true:csi_close[i+1]})
		y_predicted.append(ans[0][0])
	# checking prediction
	for j in range(sample_2):
		ans1 = sess.run(z_2,{input_x:training_set[j].reshape(1,7),y_true:csi_close[sample_8 + j+1]})
		y_predicted1.append(ans1[0][0])


mse = np.sum(np.square(csi_close[1:sample_8+1] - y_predicted))/len(y_predicted) ## 49926

np.savetxt('MLPpredict.csv',y_predicted1,delimiter = ',')

print('MSE training',mse)
print('MAPE training', np.mean(abs((csi_close[1:sample_8+1] - y_predicted))/ csi_close[1:sample_8+1]))

print('MSE testing',np.mean(np.square(csi_close[sample_8+1: 1 + sample_8+sample_2] - y_predicted1)))
print('MAPE testing', np.mean(abs((csi_close[sample_8+1: 1 + sample_8+sample_2] - y_predicted1))/ csi_close[sample_8+1: 1 + sample_8+sample_2]))
