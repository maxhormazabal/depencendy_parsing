import os
os.chdir("/content/drive/MyDrive/MASTER")
from nlu_preprocessing_utils import *

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense
import pandas as pd
import numpy as np
import math
import time
import json
import io
from conllu import parse

def getDataSourceMerged(set_type,stack_len,folder='nlu_data'):
  #set_type : 'train','test','val'
  #Getting data from files
  x = np.load(folder+'/x_'+set_type+'.npy',allow_pickle=True)
  action = np.load(folder+'/action_'+set_type+'.npy',allow_pickle=True)
  deprel = np.load(folder+'/deprel_'+set_type+'.npy',allow_pickle=True)
  x_toFloat = transformMergedDataSource(x,stack_len)
  return x_toFloat,action,deprel

def transformMergedDataSource(array,stack_len):
  #Return the same arrays but as float arrays in both dimensions
  vectorsRows = array.shape[0]
  vectorSets = array.shape[1]
  vectorElements = stack_len

  empty_data = np.zeros((vectorsRows,vectorSets,vectorElements),dtype=np.float32)

  for i,set in enumerate(array):
    for j,element in enumerate(set):
      empty_data[i][j] = np.asarray(element).astype(np.float32)
  return empty_data
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

def buildModelB(stack_len,buffer_len,action_shape,deprel_shape,optimizer='adam',learning_rate=0.01,embedd_output=50,loss='categorical_crossentropy'):
  input1 = tf.keras.layers.Input(shape=(stack_len,),name = 'Stack_Input')
  input2 = tf.keras.layers.Input(shape=(buffer_len,),name = 'Buffer_Input')

  embedding_layer = tf.keras.layers.Embedding(10000, embedd_output,mask_zero=True)
  input1_embedded = embedding_layer(input1)
  input2_embedded = embedding_layer(input2)

  # Concatenamos a lo largo del último eje
  merged = tf.keras.layers.Concatenate(axis=1,name = 'Concat_Layer')([input1_embedded, input2_embedded])
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

def getTokenizer(folder):
  path = folder + '/tokenizer.json'
  with open(path) as json_file:
      data = json.load(json_file)

  tokenizer = keras.preprocessing.text.tokenizer_from_json(json.dumps(data))
  n_token = len(tokenizer.word_index)

  print("Tokenizer with ",n_token," word sucessfully obtained.")

  return (tokenizer,n_token)

def findValueIndex(array,element):
  value = np.where(array == element)[0][0]
  return value

def predTestValues(stack,buffer,children,model,tokenizer):
  stack_len = len(stack)
  buffer_len = len(buffer)

  stack_vector = np.array([stackToVector(stack,stack_len)],dtype=np.float32)
  buffer_vector = np.array([bufferToVector(buffer,buffer_len)],dtype=np.float32)


  id_stack = [0]
  id_buffer = [*range(1, len(buffer)+1)]

  pred_action = model.predict([stack_vector,buffer_vector],verbose=0)[0][0]
  deprel = np.argmax(model.predict([stack_vector,buffer_vector],verbose=0)[1][0])

  for position in range(4):
    action = pred_action.argsort()[-(position+1)]
    if action == 3: #Shift
      break
    if action == 0: #Left
      if (stack[-1] != tokenizer.texts_to_sequences(['root'])[0][0] and id_stack[-1] not in children):
        children.append(id_stack[-1])
        break
    if action == 1: #Right
      if (id_buffer[0] not in children):
        children.append(id_buffer[0])
        break
    if action == 2: #Reduce
      if (id_stack[-1] in children):
        break

  return (action,deprel)

def getNumber2Deprel():

    number2deprel = {1: 'None',
    2: 'acl',
    3: 'acl:relcl',
    4: 'advcl',
    5: 'advmod',
    6: 'amod',
    7: 'appos',
    8: 'aux',
    9: 'aux:pass',
    10: 'case',
    11: 'cc',
    12: 'ccomp',
    13: 'compound',
    14: 'compound:prt',
    15: 'conj',
    16: 'cop',
    17: 'csubj',
    18: 'csubj:pass',
    19: 'dep',
    20: 'det',
    21: 'det:predet',
    22: 'discourse',
    23: 'dislocated',
    24: 'expl',
    25: 'fixed',
    26: 'flat',
    27: 'flat:foreign',
    28: 'iobj',
    29: 'mark',
    30: 'nmod',
    31: 'nmod:npmod',
    32: 'nmod:poss',
    33: 'nmod:tmod',
    34: 'nsubj',
    35: 'nsubj:pass',
    36: 'nummod',
    37: 'obj',
    38: 'obl',
    39: 'orphan',
    40: 'parataxis',
    41: 'punct',
    42: 'root',
    43: 'vocative',
    44: 'xcomp'}

    return number2deprel

def testingPredictions(stack_len,buffer_len,df_row,tokenizer,model,number2deprelTest):
  children = []

  # re-write variables including root at the begining
  form = np.concatenate((['root'],df_row['form']))
  id = np.concatenate(([int(0)],df_row['id']))
  head = np.full(len(id),999)
  deprel = np.full(len(id),'999',dtype='object')

  stack = form[0]
  buffer = form[1:]

  # First prediction
  (action,pred_deprel) = predTestValues(stackToVector(tokenizer.texts_to_sequences(['root'])[0],stack_len),
                bufferToVector(np.array(tokenizer.texts_to_sequences(buffer)).ravel(),buffer_len),
                children,
                model,
                tokenizer)

  #print(stack," | ",buffer," | ",action)

  while len(buffer) > 1:
    s = stack[-1] #setting attention in the last element on stack
    b = buffer[0] #setting attention in the first element on buffer

    # Left arc
    if (action == 0):
      children.append(id[findValueIndex(form,s)])
      stack = np.delete(stack,-1)
      head[findValueIndex(form,s)] = id[findValueIndex(form,b)]
      deprel[findValueIndex(form,s)] = number2deprelTest[pred_deprel]
      action_name = 'Left Arc'

    # Right arc
    elif (action == 1):  #if s is the father of b
      children.append(id[findValueIndex(form,b)])
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      head[findValueIndex(form,b)] = id[findValueIndex(form,s)]
      deprel[findValueIndex(form,b)] = number2deprelTest[pred_deprel]
      action_name = 'Right Arc'

    # Reduce
    elif (action == 2):
      stack = np.delete(stack,-1)
      action_name = 'Reduce'
    
    # Shift
    elif (action == 3):
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      action_name = 'Shift'

    #print(stack,stack.shape," | ",buffer,buffer.shape," | ",action_name)
    
    (action,pred_deprel) = predTestValues(stackToVector(tokenizer.texts_to_sequences(stack)[0],stack_len),
                  bufferToVector(np.array(tokenizer.texts_to_sequences(buffer)).ravel(),buffer_len),
                  children,
                  model,
                  tokenizer)
    
  head = checkHead(head[1:])
  deprel = checkDeprel(deprel[1:])
  return (head,deprel)

def checkHead(vector):
  k = 0
  for i,element in enumerate(vector):
    if element == 0:
      k = k + 1

  # Si existe más de uno 
  if k>1:
    for i,element in enumerate(vector):
      if element == 0 and k>1:
        vector[i] = 999
        k = k - 1
  for i,element in enumerate(vector):
    if element == 999 and k == 0:
      vector[i] = 0
      break
  for i,element in enumerate(vector):
    if element == 999:
      for l in range(1,len(vector)):
        if l not in vector:
          vector[i] = l
          break
  return vector


def checkDeprel(vector):
  k = 0
  for i,element in enumerate(vector):
    if element == 'root':
      k = k + 1
  if i>1:
    for i,element in enumerate(vector):
      if element == 'root':
        vector[i] = 'other'
  for i,element in enumerate(vector):
    if element == '999' and k == 0:
      vector[i] = 'root'
      break
  for i,element in enumerate(vector):
    if element == '999':
      vector[i] = 'other'
      
  return vector

def generateConlluForPrediction(stack_len,buffer_len,path,model,tokenizer,en_upo2number):
  en_test = parse(open(path+'/original_test_line.conllu',mode="r",encoding="utf-8").read())
  test_df = conlluToDatasetForDependency(en_test,en_upo2number)
  number2deprelTest = getNumber2Deprel()

  # Iterate each row of df_test, and take the head en deprel.
  for row in range(len(test_df)):
    (head,deprel) = testingPredictions(stack_len,buffer_len,
                                       test_df.iloc[row,:],
                                       tokenizer,
                                       model,number2deprelTest)
    for word in range(len(en_test[row])):
      if len(en_test[row]) > len(head):
        head = np.concatenate((head,np.full(len(en_test[row])-len(head),int(0))))
      if len(en_test[row]) > len(deprel):
        deprel = np.concatenate((deprel,np.full(len(en_test[row])-len(deprel),int(0))))
      en_test[row][word]['head'] = head[word]
      en_test[row][word]['deprel'] = deprel[word]
  return en_test

def generatePredictionFile(en_test,path):
  with io.open(path+"/predicted_test.conllu", mode='a', encoding='utf-8') as f:
    for token_list in en_test:
      serialized = token_list.serialize()
      lineas = serialized.split('\n\n')
      lineas_sin_vacias = []

      for linea in lineas:
        if linea.strip():
          lineas_sin_vacias.append(linea)
      serialized_sin_saltos_linea = '\n'.join(lineas_sin_vacias)
      serialized_sin_saltos_linea += '\n\n'
      f.write(serialized_sin_saltos_linea)

  with open(path+'/predicted_test.conllu', 'r') as f:
    contenido = f.read()

  lineas = contenido.split('\n\n')
  lineas_sin_vacias = []

  for linea in lineas:
    if linea.strip():
      lineas_sin_vacias.append(linea)

  contenido_con_un_salto_linea = '\n\n'.join(lineas_sin_vacias)

  with open(path+'/predicted_test_line.conllu', 'w') as f:
    f.write(contenido_con_un_salto_linea+'\n\n')
  print('Predicted file generated in ',path,'/predicted_test_line.conllu')

def saveModelADataUpos(pivot_name,pivot,
                   x_train_stack,x_train_buffer,x_train_stack_upos,x_train_buffer_upos,action_train, deprel_train,
                   x_test_stack,x_test_buffer,x_test_stack_upos,x_test_buffer_upos,action_test, deprel_test,
                   x_val_stack,x_val_buffer,x_val_stack_upos,x_val_buffer_upos,action_val, deprel_val,
                   stack_len,buffer_len,
                   num_tokens,
                   stopper,patience,
                   batch_size,epochs,
                   optimizer,learning_rate,
                   embedd_output=(30,50)):
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

  best_Action_Output_accuracy = 0
  best_Deprel_Output_accuracy = 0

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
    elif (pivot_name == 'embedding'):
      embedd_output = value
    elif (pivot_name == 'stack_buffer_sizes'):
      stack_len,buffer_len = value
      path = "nlu_data/"+str(stack_len)+"stack"+str(buffer_len)+"buffer"
      (x_train_stack,x_train_buffer,x_train_stack_upos,x_train_buffer_upos,action_train,deprel_train) = getDataSource('train',folder=path)
      (x_test_stack,x_test_buffer,x_test_stack_upos,x_test_buffer_upos,action_test,deprel_test) = getDataSource('test',folder=path)
      (x_val_stack,x_val_buffer,x_val_stack_upos,x_val_buffer_upos,action_val,deprel_val) = getDataSource('val',folder=path)

    arquitecture = 'A'
    start_time = time.time()

    model = buildModelFinal(stack_len,buffer_len,action_train[0].shape[0],deprel_train[0].shape[0],num_tokens,optimizer=optimizer,learning_rate=learning_rate,embedd_output=embedd_output,loss='categorical_crossentropy')

    score = fitModelFinal(x_train_stack,x_train_buffer,x_train_stack_upos,x_train_buffer_upos,action_train, deprel_train,
                          x_test_stack,x_test_buffer,x_test_stack_upos,x_test_buffer_upos,action_test, deprel_test,
                          x_val_stack,x_val_buffer,x_val_stack_upos,x_val_buffer_upos,action_val, deprel_val,
                          model,
                          stopper,patience,epochs,batch_size)
    loss,Action_Output_loss,Deprel_Output_loss,Action_Output_accuracy,Deprel_Output_accuracy = score

    end_time = time.time()

    training_time = (end_time - start_time)

    # Looking for the best model

    if (Action_Output_accuracy>best_Action_Output_accuracy):
      best_Action_Output_accuracy = Action_Output_accuracy
      bestActionModel = model

    if (Deprel_Output_accuracy>best_Deprel_Output_accuracy):
      best_Deprel_Output_accuracy = Deprel_Output_accuracy
      bestDeprelModel = model

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
    'embedd_output' : embedd_output_set,
    'early_stop_set' : early_stop_set,
    'time' : time_set,
  }

  return (pd.DataFrame(resultDict),models,bestActionModel,bestDeprelModel)
  #return (resultDict,models)


def buildModelFinal(stack_len,buffer_len,action_shape,deprel_shape,num_tokens,optimizer='adam',learning_rate=0.01,embedd_output=(30,50),loss='categorical_crossentropy'):
  embedding_stack_dim = embedd_output[0]
  embedding_buffer_dim = embedd_output[1]

  input_stack = tf.keras.layers.Input(shape=(stack_len,), name='stack_input')
  input_buffer = tf.keras.layers.Input(shape=(buffer_len,), name='buffer_input')

  input_upos_stack = tf.keras.layers.Input(shape=(stack_len,), name='upos_stack_input')
  input_upos_buffer = tf.keras.layers.Input(shape=(buffer_len,), name='upos_buffer_input')

  embedding_stack_layer = tf.keras.layers.Embedding(num_tokens+1, embedding_stack_dim,mask_zero = True)
  embedding_buffer_layer = tf.keras.layers.Embedding(num_tokens+1, embedding_buffer_dim,mask_zero = True)

  embedding_upos_stack_layer = tf.keras.layers.Embedding(num_tokens+1, embedding_stack_dim,mask_zero = True)
  embedding_upos_buffer_layer = tf.keras.layers.Embedding(num_tokens+1, embedding_buffer_dim,mask_zero = True)

  embedded_stack = embedding_stack_layer(input_stack)
  embedded_buffer = embedding_buffer_layer(input_buffer)

  embedded_upos_stack = embedding_upos_stack_layer(input_upos_stack)
  embedded_upos_buffer = embedding_upos_buffer_layer(input_upos_buffer)

  flat_stack = tf.keras.layers.Flatten()(embedded_stack)
  flat_buffer = tf.keras.layers.Flatten()(embedded_buffer)

  flat_upos_stack = tf.keras.layers.Flatten()(embedded_upos_stack)
  flat_upos_buffer = tf.keras.layers.Flatten()(embedded_upos_buffer)

  merged = tf.keras.layers.Concatenate()([flat_stack, flat_buffer,flat_upos_stack,flat_upos_buffer])

  dropout_1 = tf.keras.layers.Dropout(0.5)(merged)
  dropout_2 = tf.keras.layers.Dropout(0.5)(merged)

  dense_action = tf.keras.layers.Dense(32, activation='relu',name = "dense_action")(dropout_1)
  dense_deprel = tf.keras.layers.Dense(32, activation='relu',name = "dense_deprel")(dropout_2)
  dense_deprel2 = tf.keras.layers.Dense(16, activation='relu',name = "dense_deprel2")(dense_deprel)

  output_action = tf.keras.layers.Dense(action_shape, activation='softmax', name='output_action')(dense_action)
  output_deprel = tf.keras.layers.Dense(deprel_shape, activation='softmax', name='output_deprel')(dense_deprel2)

  model = tf.keras.Model(inputs=[input_stack, input_buffer,input_upos_stack,input_upos_buffer], outputs=[output_action, output_deprel])

  #tf.keras.utils.plot_model(model,show_shapes=True)
  
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

def fitModelFinal(x_train_stack,x_train_buffer,x_train_stack_upos,x_train_buffer_upos,action_train, deprel_train,
              x_test_stack,x_test_buffer,x_test_stack_upos,x_test_buffer_upos,action_test, deprel_test,
              x_val_stack,x_val_buffer,x_val_stack_upos,x_val_buffer_upos,action_val, deprel_val,
              model,
              stopper,patience=5,epochs=10,batch_size=128):
  callback = tf.keras.callbacks.EarlyStopping(monitor=stopper, patience=patience,restore_best_weights=True)
  model.fit([x_train_stack,x_train_buffer,x_train_stack_upos,x_train_buffer_upos],
            [action_train, deprel_train],
            epochs=epochs, batch_size=batch_size,
            callbacks=[callback],
            verbose = 0,
            validation_data=([x_val_stack,x_val_buffer,x_val_stack_upos,x_val_buffer_upos],[action_val,deprel_val]))
  score = model.evaluate([x_test_stack,x_test_buffer,x_test_stack_upos,x_test_buffer_upos],[action_test, deprel_test], verbose=0)
  return score

def predTestValuesUpos(stack,buffer,stack_upos,buffer_upos,stack_len,buffer_len,children,model,tokenizer):
  
  stack_vector = np.array([stackToVector(stack,stack_len)],dtype=np.float32)
  buffer_vector = np.array([bufferToVector(buffer,buffer_len)],dtype=np.float32)

  stack_upos_vector = np.array([stackToVector(stack_upos,stack_len)],dtype=np.float32)
  buffer_upos_vector = np.array([bufferToVector(buffer_upos,buffer_len)],dtype=np.float32)

  id_stack = [0]
  id_buffer = [*range(1, len(buffer)+1)]

  pred_action = model.predict([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],verbose=0)[0][0]
  deprel = np.argmax(model.predict([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],verbose=0)[1][0])

  for position in range(4):
    action = pred_action.argsort()[-(position+1)]
    if action == 3: #Shift
      break
    if action == 0: #Left
      if (stack[-1] != tokenizer.texts_to_sequences(['root'])[0][0] and id_stack[-1] not in children):
        break
    if action == 1: #Right
      if (id_buffer[0] not in children):
        break
    if action == 2: #Reduce
      if (id_stack[-1] in children):
        break

  return (action,deprel)

def testingPredictions(stack_len,buffer_len,df_row,tokenizer,model,number2deprelTest,nupos):
  children = []

  # re-write variables including root at the begining
  form = np.concatenate((['root'],df_row['form']))
  id = np.concatenate(([int(0)],df_row['id']))
  head = np.full(len(id)+1,999)
  deprel = np.full(len(id)+1,'999',dtype='object')
  upos = np.concatenate(([int(nupos+1)],df_row['upos']))

  stack = [form[0]]
  buffer = form[1:]
  
  stack_upos = [upos[0]]
  buffer_upos = upos[1:]

  while len(buffer) > 1:

    (action,pred_deprel) = predTestValuesUpos(np.array(tokenizer.texts_to_sequences(stack)).ravel(),
            np.array(tokenizer.texts_to_sequences(buffer)).ravel(),
            np.array(stack_upos).ravel(),
            np.array(buffer_upos).ravel(),
            stack_len,buffer_len,
            children,
            model,
            tokenizer)

    s = stack[-1] #setting attention in the last element on stack
    b = buffer[0] #setting attention in the first element on buffer

    # Left arc
    if (action == 0):
      children.append(id[findValueIndex(form,s)])
      stack = np.delete(stack,-1)
      stack_upos = np.delete(stack_upos,-1)
      head[findValueIndex(form,s)] = id[findValueIndex(form,b)]
      deprel[findValueIndex(form,s)] = number2deprelTest[pred_deprel]
      action_name = 'Left Arc'

    # Right arc
    elif (action == 1):  #if s is the father of b
      children.append(id[findValueIndex(form,b)])
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)
      head[findValueIndex(form,b)] = id[findValueIndex(form,s)]
      deprel[findValueIndex(form,b)] = number2deprelTest[pred_deprel]
      action_name = 'Right Arc'

    # Reduce
    elif (action == 2):
      stack = np.delete(stack,-1)
      stack_upos = np.delete(stack_upos,-1)
      action_name = 'Reduce'
    
    # Shift
    elif (action == 3):
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)
      action_name = 'Shift'

  head = checkHead(head[1:])
  deprel = checkDeprel(deprel[1:])
  return (head,deprel)

def checkHead(vector):
  # Contamos cuantos elementos son iguales a 0
  vector = list(vector)
  zeros = vector.count(0)

  if zeros == 0:
    # Buscamos el primer elemento con valor 999
    try:
      first_999_index = vector.index(999)
    except ValueError:
      # Si no existe ningun 999, cambiamos el primer elemento por 0
      vector[0] = 0
    else:
      # Si existe un 999, cambiamos ese elemento por 0
      vector[first_999_index] = 0

  # Si existen más de un cero, cambiamos todos los ceros excepto el primero
  if zeros > 1:
    first_zero_index = vector.index(0)
    for i in range(len(vector)):
      if vector[i] == 0 and i != first_zero_index:
        vector[i] = first_zero_index + 1

  # Reemplazamos los 999 por la posición del elemento que tiene el único valor 0
  for i in range(len(vector)):
    if vector[i] == 999:
      vector[i] = vector.index(0) + 1
  return vector

def checkDeprel(vector):
  vector = list(vector)
  # Contamos cuántos elementos del vector son iguales a 'root'
  root_count = vector.count('root')

  # Si no hay ningún elemento igual a 'root'
  if root_count == 0:
    # Buscamos el primer elemento igual a '999'
    for i, elem in enumerate(vector):
      if elem == '999':
        # Reemplazamos el elemento por 'root'
        vector[i] = 'root'
        break
    # Si no existe ningún elemento igual a '999'
    else:
      # Reemplazamos el primer elemento del vector por 'root'
      vector[0] = 'root'

  # Si hay más de un elemento igual a 'root'
  elif root_count > 1:
    # Recorremos el vector
    first_root_index = vector.index('root')
    for i, elem in enumerate(vector):
      # Si el elemento es igual a 'root' y no es el primer elemento
      if elem == 'root' and i != first_root_index:
        # Reemplazamos el elemento por 'other'
        vector[i] = 'other'

  # Recorremos el vector
  for i, elem in enumerate(vector):
    # Si el elemento es igual a '999'
    if elem == '999':
      # Reemplazamos el elemento por 'other'
      vector[i] = 'other'
  return vector

def generateConlluForPrediction(stack_len,buffer_len,path,model,tokenizer,en_upo2number):
  en_test = parse(open(path+'/original_test_line.conllu',mode="r",encoding="utf-8").read())
  test_df = conlluToDatasetForDependency(en_test,en_upo2number)
  number2deprelTest = getNumber2DeprelTrain()

  # Iterate each row of df_test, and take the head en deprel.
  for row in range(len(test_df)):
    (head,deprel) = testingPredictions(stack_len,buffer_len,
                                       test_df.iloc[row,:],
                                       tokenizer,
                                       model,number2deprelTest,
                                       len(en_upo2number))
    for word in range(len(en_test[row])):
      if len(en_test[row]) > len(head):
        head = np.concatenate((head,np.full(len(en_test[row])-len(head),int(0))))
      if len(en_test[row]) > len(deprel):
        deprel = np.concatenate((deprel,np.full(len(en_test[row])-len(deprel),int(0))))
      en_test[row][word]['head'] = head[word]
      en_test[row][word]['deprel'] = deprel[word]
  return en_test

def generatePredictionFile(en_test,path):
  with io.open(path+"/predicted_test.conllu", mode='a', encoding='utf-8') as f:
    for token_list in en_test:
      serialized = token_list.serialize()
      lineas = serialized.split('\n\n')
      lineas_sin_vacias = []

      for linea in lineas:
        if linea.strip():
          lineas_sin_vacias.append(linea)
      serialized_sin_saltos_linea = '\n'.join(lineas_sin_vacias)
      serialized_sin_saltos_linea += '\n\n'
      f.write(serialized_sin_saltos_linea)

  with open(path+'/predicted_test.conllu', 'r') as f:
    contenido = f.read()

  lineas = contenido.split('\n\n')
  lineas_sin_vacias = []

  for linea in lineas:
    if linea.strip():
      lineas_sin_vacias.append(linea)

  contenido_con_un_salto_linea = '\n\n'.join(lineas_sin_vacias)

  with open(path+'/predicted_test_line.conllu', 'w') as f:
    f.write(contenido_con_un_salto_linea+'\n\n')
  print('Predicted file generated in ',path,'/predicted_test_line.conllu')

def getNumber2DeprelTrain():

    number2deprel = {1: 'None',
    2: 'acl',
    3: 'acl:relcl',
    4: 'advcl',
    5: 'advmod',
    6: 'amod',
    7: 'appos',
    8: 'aux',
    9: 'aux:pass',
    10: 'case',
    11: 'cc',
    12: 'ccomp',
    13: 'compound',
    14: 'compound:prt',
    15: 'conj',
    16: 'cop',
    17: 'csubj',
    18: 'csubj:pass',
    19: 'dep',
    20: 'det',
    21: 'det:predet',
    22: 'discourse',
    23: 'dislocated',
    24: 'expl',
    25: 'fixed',
    26: 'flat',
    27: 'flat:foreign',
    28: 'iobj',
    29: 'mark',
    30: 'nmod',
    31: 'nmod:npmod',
    32: 'nmod:poss',
    33: 'nmod:tmod',
    34: 'nsubj',
    35: 'nsubj:pass',
    36: 'nummod',
    37: 'obj',
    38: 'obl',
    39: 'orphan',
    40: 'parataxis',
    41: 'punct',
    42: 'root',
    43: 'vocative',
    44: 'xcomp'}

    return number2deprel