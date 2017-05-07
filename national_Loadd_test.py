import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('01.12.2009-31.03.2017_All_Data_1_test.csv') 



df =df[['Adana','Ankara','Antalya','istanbul','Gaziantep','Dalaman','Holiday','weekDay','hour','day','month','year','UECM']]


forecast_col = 'UECM'

df.fillna(-9999, inplace=True)

#forecast_out = int(math.ceil(0.0005*len(df)))
#df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)



X = df.drop(['UECM'],1)

y = df['UECM']
print(y)

#X = preprocessing.scale(X)
#y = np.array(df['label'])



#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#########TRAINING THE MODEL
#clf =MLPRegressor(hidden_layer_sizes=(400,256,64,32,8), activation='relu', verbose=True, learning_rate_init=0.001, learning_rate='adaptive', max_iter=500,solver='adam')
#clf =MLPRegressor(hidden_layer_sizes=(400,256,64,32,8), activation='relu', verbose=True, learning_rate_init=0.001, learning_rate='adaptive', max_iter=500,solver='adam')
#clf =MLPRegressor(hidden_layer_sizes=(300,200,200), activation='relu', verbose=True, learning_rate_init=0.01, learning_rate='adaptive', max_iter=500,solver='adam')
#clf =MLPRegressor(hidden_layer_sizes=(500,300,200,30), activation='relu', verbose=True, learning_rate_init=0.01, learning_rate='adaptive', max_iter=500,solver='adam')
#clf.fit(X_train, y_train)
#with open('nisan_test.pickle','wb') as f:
#	pickle.dump(clf, f)


##### LOADING TRAINED MODEL
pickle_in =open('nisan_test.pickle', 'rb')
clf = pickle.load(pickle_in )

accuracy = clf.score(X_test, y_test)


df2 = pd.read_csv('nisan_test.csv') 



df2 =df2[['Adana','Ankara','Antalya','istanbul','Gaziantep','Dalaman','Holiday','weekDay','hour','day','month','year','UECM']]
df2.fillna(-9999, inplace=True)

subatX = df2.drop(['UECM'],1)
subaty = df2['UECM']


start=0
finish=720
predictionList = []
errorList = []
			 
for i in range (start,finish):

	predictDay =subatX.loc[i,:]
	predictDayNormalized= scaler.transform(predictDay)
	ourPrediction=clf.predict(predictDayNormalized)
	#ourPrediction = ourPrediction*1.03
	predictionList.append(ourPrediction)
	error = ourPrediction - subaty[i]
	absError = abs(ourPrediction - subaty[i])
	errorList.append(absError)

print(20*"#")
print(accuracy)
print(20*"#")
	
averageError = np.mean(errorList)
print(averageError)
count = list(range(0 , len(subaty)))			 
print(len(subaty))
print(len(predictionList))
print(len(count))
plt.plot(count,subaty,'b', label='Real')
plt.plot(count,predictionList,'r',label='Forecasted')
plt.xlabel("Time")
plt.ylabel("Load (MWh)")
plt.title('April 2017 Load Forecast')
plt.legend()
plt.show()

a = list(range(20000 , 40000))
plt.scatter(subaty,predictionList)
plt.plot(a,a,'r')
plt.xlabel("Real")
plt.ylabel("Forecasted")
plt.title('April 2017 Load Forecast')
plt.legend()
plt.show()

