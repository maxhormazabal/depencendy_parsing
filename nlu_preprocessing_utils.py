from conllu import parse
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from keras.preprocessing.text import Tokenizer
import io

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

def getActionDict():
  number2action = {
    0 : 'Left Arc',
    1 : 'Right Arc',
    2 : 'Reduce',
    3 : 'Shift'
  }

  action2number = {
    'Left Arc' : 0,
    'Right Arc' : 1,
    'Reduce' : 2,
    'Shift' : 3
  }
  return (number2action,action2number)

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

def reshapeData(df):
  empty_array = np.zeros((len(df),4),dtype='object')

  for (index,element) in enumerate(df):
    for (component,vector) in enumerate(element):
      empty_array[index][component] = vector
  return empty_array

def transformByOracle(df,stack_spaces,buffer_spaces,nupos):
  action_data = [np.array(0)]
  deprel_data = [np.array(0)]
  for index in range(df.shape[0]):
    df_row = df.iloc[index,:]
    (x_set,action_set,deprel_set) = oracle_simulator(df_row,stack_spaces,buffer_spaces,nupos)
    if index == 0:
      x_data = x_set
    else:
      x_data = np.append(x_data,x_set,axis = 0)
    action_data = np.append(action_data,action_set,axis=0)
    deprel_data = np.append(deprel_data,deprel_set,axis=0)

  if (stack_spaces == buffer_spaces):
    x_data = reshapeData(x_data)
  
  return (x_data,action_data[1:],deprel_data[1:])

def oracle_simulator(df_row,stack_spaces,buffer_spaces,nupos):
  text = df_row['form']
  text_id = df_row['id']
  text_head = df_row['head']
  text_deprel = df_row['deprel']
  text_upos = df_row['upos']

    # re-write variables including root at the begining
  form = np.concatenate((['root'],text))
  id = np.concatenate(([int(0)],text_id))
  head = np.concatenate(([int(0)],text_head))
  deprel = np.concatenate((['root'],text_deprel))
  upos = np.concatenate(([int(0)],text_upos))

  # Creating the stack and buffer
  stack = np.array(['root'])
  buffer = text

  stack_upos = np.array([nupos+1])
  buffer_upos = text_upos

  target_spaces = 2 # action and deprel

  # Sets
  x_set = np.array([np.zeros(stack_spaces),np.zeros(buffer_spaces),np.zeros(stack_spaces),np.zeros(buffer_spaces)],dtype=object)
  action_set = [np.array(0)]
  deprel_set = [np.array(0)]

  children = []

  i = 0
  counter = 0

  #while len(buffer) > 0 or len(buffer) != 1 or (counter == 0):
  while len(buffer) > 0:

    counter = counter + 1

    s = stack[-1] #setting attention in the last element on stack
    b = buffer[0] #setting attention in the first element on buffer

    prev_stack = stack
    prev_buffer = buffer

    prev_upos_stack = stack_upos
    prev_upos_buffer = buffer_upos

    #Checking Right Arc
    # finValueIndex search the text of the current value and it returns the positionn


    # Left arc
    if (s != 'root' and (id[findValueIndex(form,s)] not in children) and head[findValueIndex(form,s)] == id[findValueIndex(form,b)]): # if b is the father of s
      action = 0 #'Left'
      rel = deprel[findValueIndex(form,s)]
      stack = np.delete(stack,-1)
      children.append(id[findValueIndex(form,s)])

      stack_upos = np.delete(stack_upos,-1)

    # Right arc
    elif (id[findValueIndex(form,b)] not in children and head[findValueIndex(form,b)] == id[findValueIndex(form,s)]):  #if s is the father of b
      action = 1 #'Right'
      rel = deprel[findValueIndex(form,b)]
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      children.append(id[findValueIndex(form,b)])

      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)

    # Reduce
    elif (checkReduce(head,id,form,buffer,s) == True and id[findValueIndex(form,s)] in children):
      action = 2 #'Reduce'
      rel = 'None'
      stack = np.delete(stack,-1)
      stack_upos = np.delete(stack_upos,-1)
    
    # Shift
    else: # if there is no relationship
      action = 3 #'Shift'
      rel = 'None'
      stack = np.append(stack,b)
      buffer = np.delete(buffer,0)
      
      stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
      buffer_upos = np.delete(buffer_upos,0)

    stack_vector = stackToVector(prev_stack,stack_spaces)
    buffer_vector = bufferToVector(prev_buffer,buffer_spaces)
    stack_upos_vector = stackToVector(prev_upos_stack,stack_spaces)
    buffer_upos_vector = bufferToVector(prev_upos_buffer,buffer_spaces)
    action_set = np.append(action_set,[action],axis = 0)
    deprel_set = np.append(deprel_set,[rel],axis = 0)

    x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')

    if (i==0):
      x_set = np.append([x_set],[x_vector],axis = 0)
    else:
      x_set = np.append(x_set,[x_vector],axis = 0)

    i = i + 1
    
    # print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)
    # print(len(stack)," | ",len(buffer))
    # Reduce and Done

    if (len(buffer)==0):
      action = 2 #'Reduce'
      rel = 'None'
      stack_vector = stackToVector(stack,stack_spaces)
      buffer_vector = bufferToVector(buffer,buffer_spaces)
      stack_upos_vector = stackToVector(stack_upos,stack_spaces)
      buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
      action_set = np.append(action_set,[action],axis = 0)
      deprel_set = np.append(deprel_set,[rel],axis = 0)

      x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
      x_set = np.append(x_set,[x_vector],axis = 0)

      # print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

      while (len(stack)>1):
        stack = np.delete(stack,-1)
        stack_upos = np.delete(stack_upos,-1)
        if (len(stack)>1):
          action = 2 #'Reduce'
          rel = 'None'
        stack_vector = stackToVector(stack,stack_spaces)
        buffer_vector = bufferToVector(buffer,buffer_spaces)
        stack_upos_vector = stackToVector(stack_upos,stack_spaces)
        buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
        action_set = np.append(action_set,[action],axis = 0)
        deprel_set = np.append(deprel_set,[rel],axis = 0)

        x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
        x_set = np.append(x_set,[x_vector],axis = 0)

        # print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)
  return (x_set[1:len(x_set)-1],action_set[1:len(action_set)-1],deprel_set[1:len(deprel_set)-1])

def checkReduce(head,id,form,buffer,s):
  pending = 0
  for b in buffer:
    if head[findValueIndex(form,b)] == id[findValueIndex(form,s)]:
      pending = pending + 1
  return (pending == 0)

# def oracle_simulator(df_row,stack_spaces,buffer_spaces,nupos):
#   text = df_row['form']
#   text_id = df_row['id']
#   text_head = df_row['head']
#   text_deprel = df_row['deprel']
#   text_upos = df_row['upos']

#   # re-write variables including root at the begining
#   form = np.concatenate((['root'],text))
#   id = np.concatenate(([int(0)],text_id))
#   head = np.concatenate(([int(0)],text_head))
#   deprel = np.concatenate((['root'],text_deprel))
#   upos = np.concatenate(([int(0)],text_upos))

#   # Creating the stack and buffer
#   stack = np.array(['root'])
#   buffer = text

#   stack_upos = np.array([nupos+1])
#   buffer_upos = text_upos

#   target_spaces = 2 # action and deprel

#   # Sets
#   x_set = np.array([np.zeros(stack_spaces),np.zeros(buffer_spaces),np.zeros(stack_spaces),np.zeros(buffer_spaces)],dtype=object)
#   action_set = [np.array(0)]
#   deprel_set = [np.array(0)]

#   # vector = np.array(
#   #     [stackToVector(['1','1','1','1'],stack_spaces),
#   #     bufferToVector(['1','1','1','1'],buffer_spaces),
#   #     stackToVector(['1','1','1','1'],stack_spaces),
#   #     bufferToVector(['1','1','1','1'],buffer_spaces)],
#   #     dtype=object)

#   # x_set = np.append([x_set],[vector],axis = 0)

#   i = 0
#   while len(buffer) > 0:
#     s = stack[-1] #setting attention in the last element on stack
#     b = buffer[0] #setting attention in the first element on buffer

#     prev_stack = stack
#     prev_buffer = buffer

#     prev_upos_stack = stack_upos
#     prev_upos_buffer = buffer_upos

#     #Checking Right Arc
#     # finValueIndex search the text of the current value and it returns the position
    
#     if (head[findValueIndex(form,b)] == id[findValueIndex(form,s)]):  #if s is the father of b
#       action = 1
#       rel = deprel[findValueIndex(form,b)]
#       stack = np.append(stack,b)
#       buffer = np.delete(buffer,0)

#       stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
#       buffer_upos = np.delete(buffer_upos,0)
    
#     elif (head[findValueIndex(form,s)] == id[findValueIndex(form,b)]): # if b is the father of s
#       action = 2
#       rel = deprel[findValueIndex(form,s)]
#       stack = np.delete(stack,-1)

#       stack_upos = np.delete(stack_upos,-1)

#     else: # if there is no relationship
#       action = 3
#       rel = 'None'
#       stack = np.append(stack,b)
#       buffer = np.delete(buffer,0)
      
#       stack_upos = np.append(stack_upos,upos[findValueIndex(form,b)])
#       buffer_upos = np.delete(buffer_upos,0)

#     stack_vector = stackToVector(prev_stack,stack_spaces)
#     buffer_vector = bufferToVector(prev_buffer,buffer_spaces)
#     stack_upos_vector = stackToVector(prev_upos_stack,stack_spaces)
#     buffer_upos_vector = bufferToVector(prev_upos_buffer,buffer_spaces)
#     action_set = np.append(action_set,[action],axis = 0)
#     deprel_set = np.append(deprel_set,[rel],axis = 0)

#     x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')

#     if (i==0):
#       x_set = np.append([x_set],[x_vector],axis = 0)
#     else:
#       x_set = np.append(x_set,[x_vector],axis = 0)

#     i = i + 1
    
#     #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

#     # Reduce and Done

#     if (len(buffer)==0):
#       action = 4
#       rel = 'None'
#       stack_vector = stackToVector(stack,stack_spaces)
#       buffer_vector = bufferToVector(buffer,buffer_spaces)
#       stack_upos_vector = stackToVector(stack_upos,stack_spaces)
#       buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
#       action_set = np.append(action_set,[action],axis = 0)
#       deprel_set = np.append(deprel_set,[rel],axis = 0)

#       x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
#       x_set = np.append(x_set,[x_vector],axis = 0)

#       #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)

#       while (len(stack)>1):
#         stack = np.delete(stack,-1)
#         stack_upos = np.delete(stack_upos,-1)
#         if (len(stack)>1):
#           action = 4
#           rel = 'None'
#         elif(len(stack)==1):
#           action = 5
#           rel = 'None'
#         stack_vector = stackToVector(stack,stack_spaces)
#         buffer_vector = bufferToVector(buffer,buffer_spaces)
#         stack_upos_vector = stackToVector(stack_upos,stack_spaces)
#         buffer_upos_vector = bufferToVector(buffer_upos,buffer_spaces)
#         action_set = np.append(action_set,[action],axis = 0)
#         deprel_set = np.append(deprel_set,[rel],axis = 0)

#         x_vector = np.array([stack_vector,buffer_vector,stack_upos_vector,buffer_upos_vector],dtype='object')
#         x_set = np.append(x_set,[x_vector],axis = 0)

#         #print(stack_vector," | ",buffer_vector," | ",action," | ",rel," | ",stack_upos_vector," | ",buffer_upos_vector)
#   return (x_set[1:],action_set[1:],deprel_set[1:])

  
def applyTokenizer(dataframe,stack_len,buffer_len,tokenizer):
  df = np.copy(dataframe)
  position = 0
  for row in range(df.shape[0]):
    for index in range(stack_len):
      if (df[row][position][index]=='0'):
        df[row][position][index] = int(0)
      elif (df[row][position][index]==0):
        pass
      else:
        df[row][position][index] = tokenizer.texts_to_sequences([df[row][position][index]])[0][0]

  position = 1
  for row in range(df.shape[0]):
    for index in range(buffer_len):
      if (df[row][position][index]=='0'):
        df[row][position][index] = int(0)
      elif (df[row][position][index]==0):
        pass
      else:
        df[row][position][index] = tokenizer.texts_to_sequences([df[row][position][index]])[0][0]
  return df

def getDeprelDict(df_deprel):
  unique_deprel = np.unique(df_deprel)

  deprel2number = {}
  number2deprel = {}

  for index,element in enumerate(unique_deprel):
    deprel2number[element] = index
    number2deprel[index] = element
  return (number2deprel,deprel2number)

def deprelToNumerical(df_deprel):
  number2deprel,deprel2number = getDeprelDict(df_deprel)
  numericDeprel = np.zeros(len(df_deprel))

  for index,element in enumerate(df_deprel): 
    numericDeprel[index] =  deprel2number[element]
  return (numericDeprel,number2deprel,deprel2number)


def projectiveArcs(df):
  positions = []
  positions.extend(range(len(df)))

  for sentence in range(len(df)):
    arcs = []
    for word in range(len(df['id'][sentence])):
      subarc = (df['head'][sentence][word],df['id'][sentence][word])
      arcs.append(subarc)

    for (i,j) in arcs:
      for (k,l) in arcs:
        if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
          positions.remove(sentence) if sentence in positions else 1
  return positions

def preprocessingOneStep(base_url,file_basename,stack_len,buffer_len):
  (en_train,en_test,en_val) = readConlluDataset(base_url,file_basename)
  en_upo2number, en_number2upo, en_nupos = getUposList(en_train)
  number2action,action2number = getActionDict()

  train_df = conlluToDatasetForDependency(en_train,en_upo2number)
  test_df = conlluToDatasetForDependency(en_test,en_upo2number)
  val_df = conlluToDatasetForDependency(en_val,en_upo2number)

  train_df = train_df.iloc[projectiveArcs(train_df)]
  test_df = test_df.iloc[projectiveArcs(test_df)]
  val_df = val_df.iloc[projectiveArcs(val_df)]

  text = "root"

  for sentence in train_df['form']:
    for word in sentence:
      text = text + " " + word

  tokenizer = Tokenizer(oov_token="<OOV>",filters="") 
  tokenizer.fit_on_texts([text])
  word_index = tokenizer.word_index

  (x_train,action_train,deprel_train) = transformByOracle(train_df,stack_len,buffer_len,en_nupos)
  (x_test,action_test,deprel_test) = transformByOracle(test_df,stack_len,buffer_len,en_nupos)
  (x_val,action_val,deprel_val) = transformByOracle(val_df,stack_len,buffer_len,en_nupos)

  x_train_token = applyTokenizer(x_train,stack_len,buffer_len,tokenizer)
  x_test_token = applyTokenizer(x_test,stack_len,buffer_len,tokenizer)
  x_val_token = applyTokenizer(x_val,stack_len,buffer_len,tokenizer)

  deprel_train,number2deprel_train,deprel2number_train = deprelToNumerical(deprel_train)
  deprel_test,number2deprel_test,deprel2number_test = deprelToNumerical(deprel_test)
  deprel_val,number2deprel_val,deprel2number_val = deprelToNumerical(deprel_val)

  action_encod_train = tf.keras.utils.to_categorical(action_train)
  deprel_encod_train = tf.keras.utils.to_categorical(deprel_train)
  action_encod_test = tf.keras.utils.to_categorical(action_test)
  deprel_encod_test = tf.keras.utils.to_categorical(deprel_test)
  action_encod_val = tf.keras.utils.to_categorical(action_val)
  deprel_encod_val = tf.keras.utils.to_categorical(deprel_val)

  max_len = max([deprel_encod_train.shape[1],deprel_encod_test.shape[1],deprel_encod_val.shape[1]])

  deprel_encod_train = tf.keras.utils.pad_sequences(deprel_encod_train,maxlen=max_len,padding='post')
  deprel_encod_test = tf.keras.utils.pad_sequences(deprel_encod_test,maxlen=max_len,padding='post')
  deprel_encod_val = tf.keras.utils.pad_sequences(deprel_encod_val,maxlen=max_len,padding='post')

  folder_name = str(stack_len)+"stack"+str(buffer_len)+"buffer"
  path = "nlu_data/"+folder_name

  return(path,x_train_token,action_encod_train,deprel_encod_train,x_test_token,action_encod_test,deprel_encod_test,x_val_token,action_encod_val,deprel_encod_val)

def saveData(path,x_train_token,action_encod_train,deprel_encod_train,x_test_token,action_encod_test,deprel_encod_test,x_val_token,action_encod_val,deprel_encod_val):
  # Saving train data
  np.save(path+'/x_train.npy', x_train_token) 
  np.save(path+'/action_train.npy', action_encod_train)
  np.save(path+'/deprel_train.npy', deprel_encod_train)

  # Saving test data
  np.save(path+'/x_test.npy', x_test_token) 
  np.save(path+'/action_test.npy', action_encod_test)
  np.save(path+'/deprel_test.npy', deprel_encod_test)

  # Saving val data
  np.save(path+'/x_val.npy', x_val_token) 
  np.save(path+'/action_val.npy', action_encod_val)
  np.save(path+'/deprel_val.npy', deprel_encod_val)
  
  print("Data sucessfully saved on ./",path)

def generateConlluForTesting():
  base_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-ParTUT/master/'
  file_basename = 'en_partut-ud'
  (en_train,en_test,en_val) = readConlluDataset(base_url,file_basename)
  en_upo2number, en_number2upo, en_nupos = getUposList(en_train)

  test_df = conlluToDatasetForDependency(en_test,en_upo2number)

  # Delete UPOS = _
  for (i,token_list) in enumerate(en_test):
    for (j,dictionary) in enumerate(token_list):
      if dictionary['upos'] == "_":
        en_test[i][j].clear()

  list_projective = projectiveArcs(test_df)
  list_projective.remove(34)

  # Deleting non-projective
  for (i,token_list) in enumerate(en_test):
    if i not in list_projective:
      en_test[i].clear()

  with io.open("nlu_data/original_test.conllu", mode='a', encoding='utf-8') as f:
    for token_list in en_test:
      serialized = token_list.serialize()
      # Dividimos el contenido del archivo en una lista de líneas
      lineas = serialized.split('\n\n')
      # Creamos una nueva lista para almacenar las líneas no vacías
      lineas_sin_vacias = []

      # Recorremos la lista de líneas
      for linea in lineas:
        # Si la línea no es vacía, la añadimos a la lista
        if linea.strip():
          lineas_sin_vacias.append(linea)
      # Unimos las líneas de la lista en una cadena de texto, insertando solo un salto de línea entre cada línea
      serialized_sin_saltos_linea = '\n'.join(lineas_sin_vacias)
      serialized_sin_saltos_linea += '\n\n'
      f.write(serialized_sin_saltos_linea)

  with open('nlu_data/original_test.conllu', 'r') as f:
  # Leemos todo el contenido del archivo en una variable
    contenido = f.read()

  lineas = contenido.split('\n\n')

  lineas_sin_vacias = []

  # Recorremos la lista de líneas
  for linea in lineas:
    # Si la línea no es vacía, la añadimos a la lista
    if linea.strip():
      lineas_sin_vacias.append(linea)

  contenido_con_un_salto_linea = '\n\n'.join(lineas_sin_vacias)

  with open('nlu_data/original_test_line.conllu', 'w') as f:
    # Escribimos el contenido con solo un salto de línea en el archivo
    f.write(contenido_con_un_salto_linea+'\n\n')
  print('Original file generated in nlu_data/original_test_line.conllu')