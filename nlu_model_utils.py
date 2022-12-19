import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense
import pandas as pd
import numpy as np
import math
import time

def getDataSource(set_type,folder='nlu_data'):
  #set_type : 'train','test','val'
  #Getting data from files
  x = np.load(folder+'/x_'+set_type+'.npy',allow_pickle=True)
  action = np.load(folder+'/action_'+set_type+'.npy',allow_pickle=True)
  deprel = np.load(folder+'/deprel_'+set_type+'.npy',allow_pickle=True)

  #Using pandas to create a dataframe with features
  x_df = pd.DataFrame(x,columns = ['Stack','Buffer','Stack_UPOS','Buffer_UPOS'])

  x_stack = transformDataSource(x_df['Stack'])
  x_buffer = transformDataSource(x_df['Buffer'])
  x_stack_upos = transformDataSource(x_df['Stack_UPOS'])
  x_buffer_upos = transformDataSource(x_df['Buffer_UPOS'])
  print(x_stack.shape,x_buffer.shape,x_stack_upos.shape,x_buffer_upos.shape,action.shape,deprel.shape)
  return (x_stack,x_buffer,x_stack_upos,x_buffer_upos,action,deprel)

def transformDataSource(df_column):
  #Return the same arrays but as float arrays in both dimensions
  vectorsRows = len(df_column)
  vectorCols = len(df_column[0])

  empty_data = np.zeros((vectorsRows,vectorCols),dtype=np.float32)

  for index,element in enumerate(df_column):
    empty_data[index] = np.asarray(element).astype(np.float32)

  return empty_data

def stackToVector(prev_stack,stack_spaces):
  if (len(prev_stack) == stack_spaces):
    stack_vector = prev_stack
  elif (len(prev_stack) > stack_spaces):
    stack_vector = prev_stack[-stack_spaces:]
  elif (len(prev_stack) < stack_spaces):
    stack_vector = np.concatenate((np.full(stack_spaces-len(prev_stack),int(0)),prev_stack))
  return np.array(stack_vector,dtype='object')
  
def bufferToVector(prev_buffer,buffer_spaces):
  if (len(prev_buffer) == buffer_spaces):
    buffer_vector = prev_buffer
  elif (len(prev_buffer) > buffer_spaces):
    buffer_vector = prev_buffer[:buffer_spaces]
  elif (len(prev_buffer) < buffer_spaces):
    buffer_vector = np.concatenate((prev_buffer,np.full(buffer_spaces-len(prev_buffer),int(0))))
  return np.array(buffer_vector,dtype='object')

def buildModelA(stack_len,buffer_len,action_shape,deprel_shape,optimizer='adam',learning_rate=0.01,embedd_output=50,loss='categorical_crossentropy'):
  input1 = tf.keras.layers.Input(shape=(stack_len,),name = 'Stack_Input')
  input2 = tf.keras.layers.Input(shape=(buffer_len,),name = 'Buffer_Input')

  embedding_layer = tf.keras.layers.Embedding(10000, embedd_output,mask_zero=True)
  input1_embedded = embedding_layer(input1)
  input2_embedded = embedding_layer(input2)

  lstm1 = tf.keras.layers.LSTM(embedd_output,return_sequences=False,name="LSTM_Layer1")(input1_embedded)
  lstm2 = tf.keras.layers.LSTM(embedd_output,return_sequences=False,name="LSTM_Layer2")(input2_embedded)

  # Concatenamos a lo largo del último eje
  merged = tf.keras.layers.Concatenate(axis=1,name = 'Concat_Layer')([lstm1, lstm2])
  dense1 = tf.keras.layers.Dense(50, activation='sigmoid', use_bias=True,name = 'Dense_Layer1')(merged)
  dense2 = tf.keras.layers.Dense(15, input_dim=1, activation='relu', use_bias=True,name = 'Dense_Layer2')(dense1)
  dense3 = tf.keras.layers.Dense(30, input_dim=1, activation='relu', use_bias=True,name = 'Dense_Layer3')(dense2)
  output1 = tf.keras.layers.Dense(action_shape, activation='softmax', use_bias=True,name = 'Action_Output')(dense3)
  output2 = tf.keras.layers.Dense(deprel_shape, activation='softmax', use_bias=True,name = 'Deprel_Output')(dense3)

  model = tf.keras.Model(inputs=[input1,input2],outputs=[output1,output2])
  
  if(optimizer.lower() == 'adam'):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif(optimizer.lower() == 'sgd'):
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif(optimizer.lower() == 'rmsprop'):
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
  elif (optimizer.lower() == 'adamw'):
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
  elif (optimizer.lower() == 'adadelta'):
    opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
  elif (optimizer.lower() == 'adagrad'):
    opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
  elif (optimizer.lower() == 'adamax'):
    opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
  elif (optimizer.lower() == 'adafactor'):
    opt = tf.keras.optimizers.Adafactor(learning_rate=learning_rate)
  elif (optimizer.lower() == 'nadam'):
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
  elif (optimizer.lower() == 'ftrl'):
    opt = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
  else:
    print('Optimizer not properly defined')

  model.compile(loss=loss,optimizer=opt,metrics=['accuracy'])
  return(model)

def fitModel(x_train_stack,x_train_buffer,action_train, deprel_train,
              x_val_stack,x_val_buffer,action_val,deprel_val,
              x_test_stack,x_test_buffer,action_test,deprel_test,
              model,
              stopper,patience=3,epochs=10,batch_size=128):
  callback = tf.keras.callbacks.EarlyStopping(monitor=stopper, patience=patience,restore_best_weights=True)
  model.fit([x_train_stack,x_train_buffer],
            [action_train, deprel_train],
            epochs=epochs, batch_size=batch_size,
            callbacks=[callback],
            verbose = 0,
            validation_data=([x_val_stack,x_val_buffer],[action_val,deprel_val]))
  score = model.evaluate([x_test_stack,x_test_buffer],[action_test, deprel_test], verbose=0)
  return score

def saveModelAData(pivot_name,pivot,
                   x_train_stack,x_train_buffer,action_train, deprel_train,
                   x_val_stack,x_val_buffer,action_val,deprel_val,
                   x_test_stack,x_test_buffer,action_test,deprel_test,
                   stack_len,buffer_len,
                   stopper,patience,
                   batch_size,epochs,
                   optimizer,learning_rate,
                   embedd_output=50):
  # Creating empty lists
  arquitecture_set = []
  stack_set = []
  buffer_set = []
  action_accuracy_set = []
  deprel_accuracy_set = []
  action_loss_set = []
  deprel_loss_set = []
  batch_size_set = []
  epochs_set = []
  optimizer_set = []
  learning_rate_set = []
  embedd_output_set = []
  early_stop_set = []
  time_set = []

  models = []


  for (index,value) in enumerate(pivot):

    print('Starting execution where ',pivot_name,' varies, now with the value(s) ',value,'.')

    if (pivot_name == 'batch_size'):
      batch_size = value
    elif (pivot_name == 'epochs'):
      epochs = value
    elif (pivot_name == 'early_stop'):
      (stopper,patience) = value
    elif (pivot_name == 'optimizer'):
      (optimizer,learning_rate) = value

    arquitecture = 'A'
    start_time = time.time()

    model = buildModelA(stack_len,buffer_len,action_train[0].shape[0],deprel_train[0].shape[0],optimizer=optimizer,learning_rate=learning_rate,embedd_output=embedd_output,loss='categorical_crossentropy')

    score = fitModel(x_train_stack,x_train_buffer,action_train, deprel_train,
                  x_val_stack,x_val_buffer,action_val,deprel_val,
                  x_test_stack,x_test_buffer,action_test,deprel_test,
                  model,
                  stopper,patience,epochs,batch_size)
    loss,Action_Output_loss,Deprel_Output_loss,Action_Output_accuracy,Deprel_Output_accuracy = score

    end_time = time.time()

    training_time = (end_time - start_time)

    # Saving models
    models.append(model)

    # Append values
    arquitecture_set.append(arquitecture)
    stack_set.append(stack_len)
    buffer_set.append(buffer_len)
    action_accuracy_set.append(Action_Output_accuracy)
    deprel_accuracy_set.append(Deprel_Output_accuracy)
    action_loss_set.append(Action_Output_loss)
    deprel_loss_set.append(Deprel_Output_loss)
    batch_size_set.append(batch_size)
    epochs_set.append(epochs)
    optimizer_set.append(optimizer)
    learning_rate_set.append(learning_rate)
    embedd_output_set.append(embedd_output)
    early_stop_set.append(stopper)
    time_set.append(training_time)

  # Data dictionary

  resultDict = {
    'arquitecture' : arquitecture_set,
    'stack' : stack_set,
    'buffer' : buffer_set,
    'action_accuracy' : action_accuracy_set,
    'deprel_accuracy' : deprel_accuracy_set,
    'action_loss' : action_loss_set,
    'deprel_loss' : deprel_loss_set,
    'batch_size' : batch_size_set,
    'epochs' : epochs_set,
    'optimizer' : optimizer_set,
    'learning_rate' : learning_rate_set,
    'embedd_output' : embedd_output,
    'early_stop_set' : early_stop_set,
    'time' : time_set,
  }

  return (pd.DataFrame(resultDict),models)