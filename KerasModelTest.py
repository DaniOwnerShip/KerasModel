import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split 
from GraphXY import plot_graph
from keras import regularizers 
 
  
DATA = pd.read_csv('./dataCS.csv', delimiter=';', na_values=['NaN'])

PREASSURE_IN =  DATA['PressureIN'].astype(float).values 
DNI_RAW =  DATA['DNIraw'].astype(float).values 
DNI_RAW = np.clip(DNI_RAW, a_min=0, a_max=None)  
DNI_REAL =  DATA['DNIreal'].astype(float).values 
DNI_REAL = np.clip(DNI_REAL, a_min=0, a_max=None)  
T_IN =  DATA['TIN'].astype(float).values 
T_OUT =  DATA['TOUT'].astype(float).values 
DELTA_T =  DATA['DeltaT'].astype(float).values
DELTA_T = np.clip(DELTA_T, a_min=0, a_max=None)
JP_FX =  DATA['JP_Fx'].astype(float).values   

# variable a predecir 
FLOW_SF = DATA['Flow'].astype(float).values.reshape(-1, 1)  
 
# variables entrada 
DATA_TRAIN = [DNI_REAL, DNI_RAW, T_IN, T_OUT, DELTA_T] 

X = np.array(DATA_TRAIN).astype(float)
X = X.T   
 
X_train, X_val, y_train, y_val = train_test_split(X, FLOW_SF, test_size=0.2, random_state=0) # change random

dataLen = len(DATA_TRAIN) 

""" def custom_activation(x):
    return keras.activations.relu(x, 1200.0) """

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(dataLen,)),  
    keras.layers.Dense(128, activation='relu' , kernel_regularizer=regularizers.l1(0.9)), 
    keras.layers.Dense(128, activation='relu' , kernel_regularizer=regularizers.l2(0.1)),  
    keras.layers.Dense(256, activation='relu' ),   
    keras.layers.Dense(1)
])

   
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])   

# command: tensorboard --logdir=./logs
tensorboard_callback = TensorBoard(log_dir="./logs") 

history = model.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[tensorboard_callback], verbose=0)   
 
loss, mae = model.evaluate(X_val, y_val, verbose=0)  
 
print('loss:', loss)
print('mae :', mae) 


index = 100  # MAX 145   
sample = X_val[index]
sample = sample.reshape(1,-1) #(n,)->(1, n)y_test

ref = y_val[index] 

# predict value
result = model.predict(sample)

diff =  result - ref  

# print results 
print("result:", result)
print("ref:", ref) 
print("diff:", diff)   
print("OUTPUT FLOW =>> {:.2f}".format(result.flatten()[0]))   
 
 

plot_graph(DATA, X, model)

 