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

#print(len(momentum10_close) , len(emove10_close) , len(oscillator10_close) , len(oscilaltor10movingAverage_close) , len(rsi_close) , len(emove30_close) , len(adl),len(dates),len(csi_close))

## data normalization
sample = len(momentum10_close)
sample_7 = math.ceil(sample *.8)
sample_15 = math.floor(sample * .2)
#sample_152 =  math.floor(sample * .15)
#print(sample == (sample_7+sample_15+sample_152))


days_taken = 64*3


#cnn_momentum10_close = data_to_image(momentum10_close[0:sample_7],days_taken)
#cnn_emove10_close = data_to_image(emove10_close[0:sample_7],days_taken)
#cnn_oscillator10_close = data_to_image(oscillator10_close[0:sample_7],days_taken)
#cnn_oscilaltor10movingAverage_close = data_to_image(oscilaltor10movingAverage_close[0:sample_7],days_taken)
#cnn_rsi_close = data_to_image(rsi_close[0:sample_7],days_taken)
#cnn_emove30_close = data_to_image(emove30_close[0:sample_7],days_taken)
#cnn_adl = data_to_image(adl[0:sample_7],days_taken)
cnn_csi_close = data_to_image(csi_close[0:sample_7],days_taken)



#image_momentum10_close = gaf_Image(cnn_momentum10_close)
#image_emove10_close = gaf_Image(cnn_emove10_close)
#image_oscillator10_close = gaf_Image(cnn_oscillator10_close)
#image_oscilaltor10movingAverage_close = gaf_Image(cnn_oscilaltor10movingAverage_close)
#image_rsi_close = gaf_Image(cnn_rsi_close)
#image_emove30_close = gaf_Image(cnn_emove30_close)
#image_adl = gaf_Image(cnn_adl)

sample2 = len(cnn_csi_close)

#input_image = np.stack([image_momentum10_close,image_emove10_close,image_oscillator10_close,image_oscilaltor10movingAverage_close,image_rsi_close,image_emove30_close,image_adl])

#input_image = image_ordering(input_image)
#image_csi_close = gaf_Image(cnn_csi_close)

#cnn_momentum10_close1 = data_to_image(momentum10_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_emove10_close1 = data_to_image(emove10_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_oscillator10_close1 = data_to_image(oscillator10_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_oscilaltor10movingAverage_close1 = data_to_image(oscilaltor10movingAverage_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_rsi_close1 = data_to_image(rsi_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_emove30_close1 = data_to_image(emove30_close[sample_7:sample_15 + sample_7],days_taken)
#cnn_adl1 = data_to_image(adl[sample_7:sample_15 + sample_7],days_taken)
#cnn_csi_close1 = data_to_image(csi_close[sample_7-days_taken:sample_15 + sample_7],days_taken)

#image_momentum10_close1 = gaf_Image(cnn_momentum10_close1)
#image_emove10_close1 = gaf_Image(cnn_emove10_close1)
#image_oscillator10_close1 = gaf_Image(cnn_oscillator10_close1)
#image_oscilaltor10movingAverage_close1 = gaf_Image(cnn_oscilaltor10movingAverage_close1)
#image_rsi_close1 = gaf_Image(cnn_rsi_close1)
#image_emove30_close1 = gaf_Image(cnn_emove30_close1)
#image_adl1 = gaf_Image(cnn_adl1)
#image_csi_close1 = gaf_Image(cnn_csi_close1)

#sample2_validation = len(cnn_csi_close1)

#validation_image = np.stack([image_momentum10_close1,image_emove10_close1,image_oscillator10_close1,image_oscilaltor10movingAverage_close1,image_rsi_close1,image_emove30_close1,image_adl1])
#validation_image = np.stack([cnn_emove10_close1,cnn_emove30_close1,cnn_oscillator10_close1])
#validation_image = image_ordering(validation_image)

#cnn_momentum10_close2 = data_to_image(momentum10_close[sample_7 + sample_15:sample_15 + sample_7 + sample_152],days_taken)
#cnn_emove10_close2 = data_to_image(emove10_close[sample_7-days_taken:sample_7 + sample_15],days_taken)
#cnn_oscillator10_close2 = data_to_image(oscillator10_close[sample_7 + sample_15:sample_15 + sample_7 + sample_152],days_taken)
#cnn_oscilaltor10movingAverage_close2 = data_to_image(oscilaltor10movingAverage_close[sample_7-days_taken:sample_7 + sample_15],days_taken)
#cnn_rsi_close2 = data_to_image(rsi_close[sample_7 + sample_15:sample_15 + sample_7 + sample_152],days_taken)
#cnn_emove30_close2 = data_to_image(emove30_close[sample_7-days_taken:sample_7 + sample_15],days_taken)
#cnn_adl2 = data_to_image(adl[sample_7 + sample_15:sample_15 + sample_7 + sample_152],days_taken)
cnn_csi_close2 = data_to_image(csi_close[sample_7 - days_taken:sample_15 + sample_7],days_taken)

sample2_testing = len(cnn_csi_close2)

#testing_image = image_ordering(testing_image)

sample = len(momentum10_close)
sample_8 = math.ceil(sample *.8)
sample_2 = math.floor(sample * .2)


cnnpredict = np.loadtxt('CNNpredict.csv')
mlppredict = np.loadtxt('MLPpredict.csv')
lstmpredict = np.loadtxt('LSTMpredict.csv')

actual = np.diff(csi_close[sample_8+1:sample_8+sample_2+1])
mlpdirection = mlppredict[1:] - csi_close[sample_8+1:sample_8+sample_2]
mlpcorrectratio = mlpdirection / actual
mlpincorrectratio = 1 + (np.abs(mlpdirection) / np.abs(actual))
direction1 = mlpdirection * actual
direction1 = mlpdirection >= 0


cnndirection = cnnpredict[1:] - csi_close[sample_8+1:sample_8+sample_2]
cnncorrectratio = cnndirection / actual
cnnincorrectratio = 1 + (np.abs(cnndirection) / np.abs(actual))
direction2 = cnndirection * actual
direction2 = direction2 >= 0

lstmdirection = lstmpredict[1:] - csi_close[sample_8+1:sample_8+sample_2]
lstmcorrectratio = lstmdirection / actual
lstmincorrectratio = 1 + (np.abs(lstmdirection) / np.abs(actual))
direction3 = lstmdirection * actual
direction3 = direction3 >= 0

print(np.sum(direction1)/len(actual))
print(np.sum(direction2)/len(actual))
print(np.sum(direction3)/len(actual))

print(np.sum(mlpcorrectratio)/len(actual))
print(np.sum(cnncorrectratio)/len(actual))
print(np.sum(lstmcorrectratio)/len(actual))

print(np.sum(mlpincorrectratio)/len(actual))
print(np.sum(cnnincorrectratio)/len(actual))
print(np.sum(lstmincorrectratio)/len(actual))


mseMLP = np.mean(np.square(csi_close[sample_8+1:sample_8+sample_2+1] - mlppredict))
mseCNN = np.mean(np.square(csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1] - cnnpredict))
mseLSTM = np.mean(np.square(csi_close[sample_8+1:sample_8+sample_2+1] - lstmpredict))

madMLP = np.mean(np.absolute(csi_close[sample_8+1:sample_8+sample_2+1] - mlppredict))
madCNN = np.mean(np.absolute(csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1] - cnnpredict))
madLSTM = np.mean(np.absolute(csi_close[sample_8+1:sample_8+sample_2+1] - lstmpredict))

stdMLP = np.std(csi_close[sample_8+1:sample_8+sample_2+1]-mlppredict)
stdCNN = np.std(csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1]-cnnpredict)
stdLSTM = np.std(csi_close[sample_8+1:sample_8+sample_2+1]-lstmpredict)


print("mad MLP",madMLP)
print("mad CNN",madCNN)
print("mad LSTM",madLSTM)

print("MSE MLP",mseMLP)
print("MSE CNN",mseCNN)
print("MSE LSTM",mseLSTM)
print('MAPE MLP', np.mean(abs((csi_close[sample_8+1: 1 + sample_8+sample_2] - mlppredict))/ csi_close[sample_8+1: 1 + sample_8+sample_2]))
print('MAPE CNN', np.mean(abs((csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1] - cnnpredict))/ csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1]))
print('MAPE LSTM', np.mean(abs((csi_close[sample_8 + 1: sample_2 + 1 + sample_8 ] - lstmpredict))/ csi_close[sample_8 + 1: 1 + sample_8 + sample_2]))

print(madCNN/madMLP,mseCNN/mseMLP,np.mean(abs((csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1] - cnnpredict))/ csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1])/np.mean(abs((csi_close[sample_8+1: 1 + sample_8+sample_2] - mlppredict))/ csi_close[sample_8+1: 1 + sample_8+sample_2]))


plt.figure(figsize=(7,4))
plt.subplots_adjust(top = 0.96,right = 0.96,left = 0.10)
plt.gca().set_xlim([np.min(dates[sample_8+1:sample_8 + sample_2+1]),np.max(dates[sample_8+1:sample_8 + sample_2+1])])
plt.plot(dates[sample_8+1:sample_8 + sample_2+1],csi_close[sample_8+1:sample_8+sample_2+1],label = 'actual',linewidth = 0.5)
plt.plot(dates[sample_8+1:sample_8 + sample_2+1],mlppredict,'r--',label = 'prediction',linewidth = 0.5)
plt.legend(loc = 'upper left',framealpha=0)
plt.ylabel("CSI 300 Price")
plt.xlabel("Time")
plt.show()
plt.figure(figsize=(7,4))
plt.subplots_adjust(top = 0.96,right = 0.96,left = 0.10)
plt.gca().set_xlim([np.min(dates[sample_8+1:sample_8 + sample_2+1]),np.max(dates[sample_8+1:sample_8 + sample_2+1])])##
plt.plot(dates[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1],csi_close[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1],label = 'actual',linewidth = 0.5)
plt.plot(dates[sample2 + days_taken+1:sample2+days_taken+ sample2_testing+1],cnnpredict,'r--',label = 'prediction',linewidth = 0.5)
plt.legend(loc = 'upper left',framealpha=0)
plt.ylabel("CSI 300 Price")#
plt.xlabel("Time")#
plt.show()
plt.figure(figsize=(7,4))
plt.subplots_adjust(top = 0.96,right = 0.96,left = 0.10)
plt.gca().set_xlim([np.min(dates[sample_8+1:sample_8 + sample_2+1]),np.max(dates[sample_8+1:sample_8 + sample_2+1])])
plt.plot(dates[sample_8+1:sample_8 + sample_2+1],csi_close[sample_8+1:sample_8+sample_2+1],label = 'actual',linewidth = 0.5)
plt.plot(dates[sample_8+1:sample_8 + sample_2+1],lstmpredict,'r--',label = 'prediction',linewidth = 0.5)
plt.legend(loc = 'upper left',framealpha=0)
plt.ylabel("CSI 300 Price")#
plt.xlabel("Time")
plt.show()