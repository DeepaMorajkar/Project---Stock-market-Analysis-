import pandas as  pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt
from datetime import timedelta
from datetime import date

from nsepy import get_history
from datetime import date
from keras.models import load_model

start_date=date(2015,1,1)
end_date=date(2023,2,15)

st.title('Stock Price Prediction')
user_input=st.text_input('Enter Stock Ticker','RELIANCE')
reliance=get_history(symbol=user_input,start=start_date,end=end_date)

st.subheader('Data from 2015 to 2023')
st.write(reliance.describe())

#visualization
st.subheader('Closing price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.plot(reliance.Close)
st.pyplot(fig)

st.subheader('Closing price Vs Time Chart with 100MA')
ma100=reliance.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.plot(ma100)
plt.plot(reliance.Close)
st.pyplot(fig)

st.subheader('Closing price Vs Time Chart with 100MA and 200MA')
ma100=reliance.Close.rolling(100).mean()
ma200=reliance.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.plot(ma100)
plt.plot(ma200)
plt.plot(reliance.Close)
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import math
# Create a new dataframe with only Close column
data=reliance.filter(['Close'])
# Convert the dataframe to a numpy array
dataset=data.values
# Get the number of rows to train the model on
training_data_len=math.ceil(len(dataset)*.8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# Create the training dataset
# Create the scaled training dataset
train_data=scaled_data[0:training_data_len,:]

#load my model
model=load_model('keras_model.h5')

# Create testing dataset
# Create a new array containing 
test_data=scaled_data[training_data_len-60:,:]
# Create the datasets x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range (60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

# Convert the data to numpy array
x_test=np.array(x_test)

# Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

# Plot the data
st.subheader('Predictions Vs Original')
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
# Visualize the data
fig2=plt.figure(figsize=(16,8))
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
st.pyplot(fig2)

st.subheader('Stock Price Prediction by Date')

df1=reliance.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

date1 = st.date_input("Enter Date in this format yyyy-mm-dd")

result = st.button("Predict")
#st.write(result)
if result:
	from datetime import datetime
	my_time = datetime.min.time()
	date1 = datetime.combine(date1, my_time)
	#date1=str(date1)
	#date1=dt.datetime.strptime(time_str,"%Y-%m-%d")

	nDay=date1-datemax
	nDay=nDay.days

	date_rng = pd.date_range(start=datemax, end=date1, freq='D')
	date_rng=date_rng[1:date_rng.size]
	lst_output=[]
	n_steps=x_input.shape[1]
	i=0

	while(i<=nDay):
    
	    if(len(temp_input)>n_steps):
        	  #print(temp_input)
        	    x_input=np.array(temp_input[1:]) 
        	    print("{} day input {}".format(i,x_input))
        	    x_input=x_input.reshape(1,-1)
        	    x_input = x_input.reshape((1, n_steps, 1))
        		#print(x_input)
        	    yhat = model.predict(x_input, verbose=0)
        	    print("{} day output {}".format(i,yhat))
        	    temp_input.extend(yhat[0].tolist())
        	    temp_input=temp_input[1:]
        	    #print(temp_input)
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	    else:
        	    x_input = x_input.reshape((1, n_steps,1))
        	    yhat = model.predict(x_input, verbose=0)
        	    print(yhat[0])
        	    temp_input.extend(yhat[0].tolist())
        	    print(len(temp_input))
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	res =scaler.inverse_transform(lst_output)
#output = res[nDay-1]

	output = res[nDay]

	st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
	st.success('The Price is {}'.format(np.round(output[0], 2)))

	#st.write("predicted price : ",output)

	predictions=res[res.size-nDay:res.size]
	print(predictions.shape)
	predictions=predictions.ravel()
	print(type(predictions))
	print(date_rng)
	print(predictions)
	print(date_rng.shape)

	@st.cache
	def convert_df(df):
   		return df.to_csv().encode('utf-8')
	df = pd.DataFrame(data = date_rng)
	df['Predictions'] = predictions.tolist()
	df.columns =['Date','Price']
	st.write(df)
	csv = convert_df(df)
	st.download_button(
   		"Press to Download",
   		csv,
  		 "file.csv",
   		"text/csv",
  		 key='download-csv'
	)

  #visualization

	fig =plt.figure(figsize=(10,6))
	xpoints = date_rng
	ypoints =predictions
	plt.xticks(rotation = 90)
	plt.plot(xpoints, ypoints)
	st.pyplot(fig)
