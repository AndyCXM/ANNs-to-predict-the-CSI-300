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


## data normalization
sample = len(momentum10_close)
sample_7 = math.ceil(sample *.8)
sample_15 = math.floor(sample * .2)



days_taken = 64*3


cnn_emove10_close = data_to_image(emove10_close[0:sample_7],days_taken)
cnn_emove30_close = data_to_image(emove30_close[0:sample_7],days_taken)
cnn_adl =data_to_image(adl[0:sample_7],days_taken)

sample2 = len(cnn_emove10_close)
input_image = np.stack([cnn_emove10_close,cnn_emove30_close,cnn_adl])
input_image = np.transpose(input_image,[1,2,3,0])

print(input_image.shape)


cnn_emove10_close2 = data_to_image(emove10_close[sample_7 - days_taken:sample_15 + sample_7],days_taken)
cnn_emove30_close2 = data_to_image(emove30_close[sample_7 - days_taken:sample_15 + sample_7],days_taken)
cnn_adl2 = data_to_image(adl[sample_7 - days_taken:sample_15 + sample_7],days_taken)


sample2_testing = len(cnn_emove10_close2)

testing_image = np.stack([cnn_emove10_close2,cnn_emove30_close2,cnn_adl2])
testing_image = np.transpose(testing_image,[1,2,3,0])

def weight_variable(shape):
	initial = tf.keras.initializers.he_normal()
	return tf.Variable(initial(shape))

def bias_variable(shape):
	initial = tf.keras.initializers.he_normal()
	return tf.Variable(initial(shape))

def conv2d(x, W,mode):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=mode)

def conv1d(x, W,mode):
	return tf.nn.conv1d(x, W, stride=1, padding=mode)

def pool(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

def conv_layer(input,W,b,mode = 'SAME'):
	return tf.nn.leaky_relu(conv2d(input, W,mode) + b)

def conv_layer1d(input,W,b,mode = 'SAME'):
	return tf.nn.leaky_relu(conv1d(input, W,mode) + b)

def full_layer(input1,inputneurons, size):
	W = weight_variable([inputneurons, size])
	b = bias_variable([size])
	return tf.matmul(input1, W) + b

def weights_calc(shape):
	ans =  tf.keras.initializers.he_normal()
	ans = tf.Variable(ans(shape))
	d = tf.constant([],dtype = tf.float32)
	for i in range(shape[0]):
		for x in range(shape[1]):
			for y in range(shape[2]):
				for z in range(shape[3]):
					if z == y:
						m = tf.Variable([ans[i][x][y][z]],dtype = tf.float32)	
						d = tf.concat([d,m],axis = 0)					
					else:
						n = tf.constant([0],dtype = tf.float32)	
						d = tf.concat([d,n],axis = 0)	
	return tf.reshape(d,(shape))


learning_rate = 0.01
epochs = 50

x_input = tf.placeholder(tf.float32,shape = (None,days_taken,days_taken,3))
y_true = tf.placeholder(tf.float32,shape = (None))

with tf.device('/cpu:0'):
	initial = tf.keras.initializers.he_normal()

	shape_w1 = [3,3,3,3]
	shape_w2 = [3,3,3,3]
	shape_w3 = [3,3,4,8]
	shape_w4 = [3,3,14,7]

	w1 = tf.Variable(initial(shape_w1))
	b1 = tf.Variable(initial([shape_w1[3]]))
	conv1 = conv_layer(x_input,w1,b1)

	w2 = tf.Variable(initial(shape_w2))
	b2 = tf.Variable(initial([shape_w2[3]]))
	conv2 = conv_layer(conv1,w2,b2)
	conv2_pool = pool(conv2)
	print(conv2_pool)
	
	conv3_flat = tf.reshape(conv2_pool, [-1, 64*64*3])

	a_2 = tf.nn.leaky_relu(full_layer(conv3_flat,64*64*3,64*64))
	a_3 = tf.nn.leaky_relu(full_layer(a_2,64*64,10))
	z_2 = full_layer(a_3,10,1)
	cost_loss = tf.reduce_mean(tf.square(y_true - z_2)) 

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	train = optimizer.minimize(cost_loss)

	accuracy = tf.reduce_mean(tf.square(y_true - z_2))

batch_size = 32

y_predicted = []
y_predicted_testing = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for steps in range(epochs):
		for i in range(0,sample2,batch_size):
			sess.run(train,feed_dict = {x_input:input_image[i:i+batch_size],y_true:csi_close[i+days_taken+1:i+batch_size+days_taken+1]})
	for i in range(sample2):
		ans = sess.run(z_2,feed_dict = {x_input:input_image[i].reshape([-1,days_taken,days_taken,3])})
		y_predicted.append(ans[0][0])
	for i in range(sample2_testing):
		ans2 = sess.run(z_2,feed_dict = {x_input:testing_image[i].reshape([-1,days_taken,days_taken,3])})
		y_predicted_testing.append(ans2[0][0])

y_predicted_testing = np.array(y_predicted_testing)
np.savetxt('CNNpredict.csv',y_predicted_testing,delimiter = ',')

y_predicted = np.array(y_predicted)
mse = np.sum(np.square(csi_close[days_taken+1:sample2+days_taken +1] - y_predicted))/len(y_predicted) ## 49926#
print('MSE training',mse)
print('MAPE training', np.mean(abs((csi_close[days_taken +1:sample2+days_taken+1] - y_predicted))/ csi_close[days_taken+1:sample2+days_taken+1]))

mse1 = np.sum(np.square(csi_close[sample2+days_taken + 1:sample2+days_taken+ sample2_testing +1] - y_predicted_testing))/len(y_predicted_testing) 
print('MSE testing',mse1)
print('MAPE testing', np.mean(abs((csi_close[sample2 + days_taken +1:sample2+days_taken+ sample2_testing +1] - y_predicted_testing))/ csi_close[sample2+days_taken +1:sample2+days_taken+ sample2_testing +1]))
