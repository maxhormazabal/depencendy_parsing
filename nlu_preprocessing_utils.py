from conllu import parse
import tensorflow as tf
import pandas as pd
import numpy as np
import math

def readConlluDataset(url,file_name):
  # url 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/'
  # file_name en_ewt-ud'
  # Custom URL by type of set

  training_file = file_name + '-train.conllu'
  testing_file = file_name + '-test.conllu'
  validation_file = file_name + '-dev.conllu'

  training_url = url + training_file
  testing_url = url + testing_file
  validation_url = url + validation_file

  path_to_training = tf.keras.utils.get_file(training_file,training_url)
  path_to_testing = tf.keras.utils.get_file(testing_file,testing_url)
  path_to_validation = tf.keras.utils.get_file(validation_file,validation_url)

  training_data = parse(open(path_to_training,mode="r",encoding="utf-8").read())
  testing_data = parse(open(path_to_testing,mode="r",encoding="utf-8").read())
  validation_data = parse(open(path_to_validation,mode="r",encoding="utf-8").read())

  return (training_data,testing_data,validation_data)

def getUposList(sentences):
  total_upos = list()
  for sentence in sentences:
    for detail in sentence:
      current_upos = detail['upos']
      if (current_upos != '_') and (current_upos not in total_upos):
        total_upos.append(current_upos)
  number2upo = {}
  upo2number = {}
  for i in range(0,len(total_upos)):
    upo2number[total_upos[i]] = i+1
    number2upo[i+1] = total_upos[i]
  print("Total of different UPOS: ",len(total_upos))
  print(upo2number)
  return (upo2number,number2upo,len(total_upos))

def conlluToDatasetForDependency(sentences,upos2number):
  df = pd.DataFrame({'id' : [],'form' : [],'head' : [],'deprel' : [],'upos' : []})
  for i in range(0,len(sentences)):
      index = pd.DataFrame.from_dict(sentences[i][:])['id'].values
      form = pd.DataFrame.from_dict(sentences[i][:])['form'].values
      head = pd.DataFrame.from_dict(sentences[i][:])['head'].values
      deprel = pd.DataFrame.from_dict(sentences[i][:])['deprel'].values
      upos = pd.DataFrame.from_dict(sentences[i][:])['upos'].values
      results = np.where(deprel == "_")
      if len(results[0])>0:
          index = np.delete(index, results)
          form = np.delete(form, results)
          head = np.delete(head, results)
          deprel = np.delete(deprel, results)
          upos = np.delete(upos, results)
      numb_upos = [upos2number[upo] for upo in upos]
      new_row = pd.Series({'id' : index,'form' :form ,'head' : head,'deprel' : deprel, 'upos': numb_upos})
      df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
  return df

def findValueIndex(array,element):
  value = np.where(array == element)[0][0]
  return value


def stackToVector(prev_stack,stack_spaces):
  if (len(prev_stack) == stack_spaces):
    stack_vector = prev_stack
  elif (len(prev_stack) > stack_spaces):
    stack_vector = prev_stack[-stack_spaces:]
  elif (len(prev_stack) < stack_spaces):
    stack_vector = np.concatenate((np.full(stack_spaces-len(prev_stack),math.nan),prev_stack))
  return np.array(stack_vector)
  
def bufferToVector(prev_buffer,buffer_spaces):
  if (len(prev_buffer) == buffer_spaces):
    buffer_vector = prev_buffer
  elif (len(prev_buffer) > buffer_spaces):
    buffer_vector = prev_buffer[:buffer_spaces]
  elif (len(prev_buffer) < buffer_spaces):
    buffer_vector = np.concatenate((prev_buffer,np.full(buffer_spaces-len(prev_buffer),math.nan)))
  return np.array(buffer_vector)

def transformByOracle(df,stack_spaces,buffer_spaces):
  x_data = [np.zeros(4)]
  action_data = [np.zeros(2)]

  for index in range(0,df.shape[0]):
    df_row = df.iloc[index,:]
    (x_set,action_set) = oracle_simulator(df_row,stack_spaces,buffer_spaces)
    x_data = np.append(x_data,x_set,axis=0)
    action_data = np.append(action_data,action_set,axis=0)
  return (x_data[1:],action_data[1:])

def oracle_simulator(df_row,stack_spaces,buffer_spaces):
  text = df_row['form']
  text_id = df_row['id']
  text_head = df_row['head']
  text_deprel = df_row['deprel']
  text_upos = df_row['upos']

  # re-write variables including root at the begining
  form = np.concatenate((['root'],text))
  id = np.concatenate(([0],text_id))
  head = np.concatenate(([0],text_head))
  deprel = np.concatenate((['root'],text_deprel))
  upos = np.concatenate(([0],text_upos))

  # Creating the stack and buffer
  stack = np.array(['root'])
  buffer = text

  stack_upos = np.array([0])
  buffer_upos = text_upos

  target_spaces = 2 # action and deprel

  # Sets
  x_set = [np.zeros(4)]
  action_set = [np.zeros(target_spaces)]

  while len(buffer) > 0:
    s = stack[-1] #setting attention in the last element on stack
    b = buffer[0] #setting attention in the first element on buffer

    prev_stack = stack
    prev_buffer = buffer

    prev_upos_stack = stack_upos
    prev_upos_buffer = buffer_upos

    #Checking Right Arc
    # finValueIndex search the text of the current value and it returns the position
    
    if (head[findValueIndex(form,b)] == id[findValueIndex(form,s)]):  #if s is the father of b
      action = 'Right Arc'
      rel = deprel[findValueIndex(form,b)]
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)

      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)
    
    elif (head[findValueIndex(form,s)] == id[findValueIndex(form,b)]): # if b is the father of s
      action = 'Left Arc'
      rel = deprel[findValueIndex(form,s)]
      stack = np.delete(stack,-1)

      stack_upos = np.delete(stack_upos,-1)

    else: # if there is no relationship
      action = 'Shift'
      rel = 'None'
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      
      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)

    stack_vector = stackToVector(prev_stack,stack_spaces)
    buffer_vector = bufferToVector(prev_buffer,buffer_spaces)
    stack_upos_vector = stackToVector(prev_upos_stack,stack_spaces)
    buffer_upos_vector = bufferToVector(prev_upos_buffer,buffer_spaces)
    action_set = np.append(action_set,[[action,rel]],axis = 0)

    x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
    x_set = np.append(x_set,[x_vector],axis = 0)
    
    #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

    # Reduce and Done

    if (len(buffer)==0):
      action = 'Reduce'
      rel = 'None'
      stack_vector = stackToVector(stack,stack_spaces)
      buffer_vector = bufferToVector(buffer,buffer_spaces)
      stack_upos_vector = stackToVector(stack_upos,stack_spaces)
      buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
      action_set = np.append(action_set,[[action,rel]],axis = 0)

      x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
      x_set = np.append(x_set,[x_vector],axis = 0)

      #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

      while (len(stack)>1):
        stack = np.delete(stack,-1)
        stack_upos = np.delete(stack_upos,-1)
        if (len(stack)>1):
          action = 'Reduce'
          rel = 'None'
        elif(len(stack)==1):
          action = 'Done'
          rel = 'None'
        stack_vector = stackToVector(stack,stack_spaces)
        buffer_vector = bufferToVector(buffer,buffer_spaces)
        stack_upos_vector = stackToVector(stack_upos,stack_spaces)
        buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
        action_set = np.append(action_set,[[action,rel]],axis = 0)

        x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
        x_set = np.append(x_set,[x_vector],axis = 0)

        #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)
  return (x_set[1:],action_set[1:])


def oracle_simulator_old(df_row,stack_spaces,buffer_spaces):
  text = df_row['form']
  text_id = df_row['id']
  text_head = df_row['head']
  text_deprel = df_row['deprel']
  text_upos = df_row['upos']

  # re-write variables including root at the begining
  form = np.concatenate((['root'],text))
  id = np.concatenate(([0],text_id))
  head = np.concatenate(([0],text_head))
  deprel = np.concatenate((['root'],text_deprel))
  upos = np.concatenate(([0],text_upos))

  # Creating the stack and buffer
  stack = np.array(['root'])
  buffer = text

  stack_upos = np.array([0])
  buffer_upos = text_upos

  target_spaces = 2 # action and deprel

  # Sets
  stack_set = [np.zeros(stack_spaces)]
  stack_upos_set = [np.zeros(stack_spaces)]
  buffer_set = [np.zeros(buffer_spaces)]
  buffer_upos_set = [np.zeros(buffer_spaces)]
  action_set = [np.zeros(target_spaces)]


  while len(buffer) > 0:
    s = stack[-1] #setting attention in the last element on stack
    b = buffer[0] #setting attention in the first element on buffer

    prev_stack = stack
    prev_buffer = buffer

    prev_upos_stack = stack_upos
    prev_upos_buffer = buffer_upos

    #Checking Right Arc
    # finValueIndex search the text of the current value and it returns the position
    
    if (head[findValueIndex(form,b)] == id[findValueIndex(form,s)]):  #if s is the father of b
      action = 'Right Arc'
      rel = deprel[findValueIndex(form,b)]
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)

      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)
    
    elif (head[findValueIndex(form,s)] == id[findValueIndex(form,b)]): # if b is the father of s
      action = 'Left Arc'
      rel = deprel[findValueIndex(form,s)]
      stack = np.delete(stack,-1)

      stack_upos = np.delete(stack_upos,-1)

    else: # if there is no relationship
      action = 'Shift'
      rel = 'None'
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      
      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)

    stack_vector = stackToVector(prev_stack,stack_spaces)
    buffer_vector = bufferToVector(prev_buffer,buffer_spaces)
    action_set = np.append(action_set,[[action,rel]],axis = 0)
    stack_set = np.append(stack_set,[stack_vector],axis = 0)
    buffer_set = np.append(buffer_set,[buffer_vector],axis = 0)

    stack_upos_vector = stackToVector(prev_upos_stack,stack_spaces)
    buffer_upos_vector = bufferToVector(prev_upos_buffer,buffer_spaces)
    stack_upos_set = np.append(stack_upos_set,[stack_upos_vector],axis = 0)
    buffer_upos_set = np.append(buffer_upos_set,[buffer_upos_vector],axis = 0)

    #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

    # Reduce and Done

    if (len(buffer)==0):
      action = 'Reduce'
      rel = 'None'
      stack_vector = stackToVector(stack,stack_spaces)
      buffer_vector = bufferToVector(buffer,buffer_spaces)
      action_set = np.append(action_set,[[action,rel]],axis = 0)
      stack_set = np.append(stack_set,[stack_vector],axis = 0)
      buffer_set = np.append(buffer_set,[buffer_vector],axis = 0)

      stack_upos_vector = stackToVector(stack_upos,stack_spaces)
      buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
      stack_upos_set = np.append(stack_upos_set,[stack_upos_vector],axis = 0)
      buffer_upos_set = np.append(buffer_upos_set,[buffer_upos_vector],axis = 0)

      #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

      while (len(stack)>1):
        stack = np.delete(stack,-1)
        stack_upos = np.delete(stack_upos,-1)
        if (len(stack)>1):
          action = 'Reduce'
          rel = 'None'
        elif(len(stack)==1):
          action = 'Done'
          rel = 'None'
        stack_vector = stackToVector(stack,stack_spaces)
        buffer_vector = bufferToVector(buffer,buffer_spaces)
        action_set = np.append(action_set,[[action,rel]],axis = 0)
        stack_set = np.append(stack_set,[stack_vector],axis = 0)
        buffer_set = np.append(buffer_set,[buffer_vector],axis = 0)

        stack_upos_vector = stackToVector(stack_upos,stack_spaces)
        buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
        stack_upos_set = np.append(stack_upos_set,[stack_upos_vector],axis = 0)
        buffer_upos_set = np.append(buffer_upos_set,[buffer_upos_vector],axis = 0)

        #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)
  return ([stack_set[1:],buffer_set[1:],stack_upos_set[1:],buffer_upos_set[1:]],action_set[1:])