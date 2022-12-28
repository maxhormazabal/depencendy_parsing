# P2 - User manual

**Students:** Maximiliano Hormazábal - Mutaz Abueisheh

This project is available on GitHub. [Click here to go to the repository](https://github.com/maxhormazabal/depencendy_parsing)

### 0. Identify the files:

This project have two main files:
- `p2_preprocessing.ipynb`: Is the one you have to use to know all the steps for get the raw data, make the transformations and load the final data set to use it after.
- `p2_ann_models.ipynb`: Here you have all the model's arquitectures, in this program we do not transform the data it is just for experiments and get results.

### 1. Install/Import libraries

In this project files and folders are going to be created in some steps. If you are going to work on Google Colab it is crucial to connect your notebook to your Google Drive profile for files/folders managment doing the following:

```python
# Getting access to Google Drive files
from google.colab import drive
drive.mount('/content/drive')
```

The first step is to import the libraries we are going to use (or install if is necessary), the most important are:
- Conllu to read the languages files correctly (in this case we will use parse module)
- Tensorflow to work with ANN, Tokenizer and other Machine Learning Tools
- Pandas to create/transform dataframes
- Numpy to work with numbers and some data structures

To install you can use `!pip install <name>` to install the specific tool that you need.

As following:

```python
# Libraries
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
import pandas as pd
import numpy as np
import math
import time

# Setting working directory and importing functions
import os
os.chdir("/content/drive/MyDrive/MASTER") # <- Folder where you saved the utils
from nlu_model_utils import *
```

## Preprocessing Notebook

### 2. Download the langagues datasets

To find the dataset that you want to use visit this <a href="https://github.com/UniversalDependencies/" target="_blank">Github Repository</a>, there are a many posibilities of languague. If you are looking for an specific language the "Repositories" search bar is useful to it. 
    
<img src="https://raw.githubusercontent.com/maxhormazabal-test/nlu-p1/main/1.png" width="700">

When you are ready do the following:
- Go to the repository of your selected language
- As you see you have 3 files: training, testing, validation (train, test, dev)

For instance

<img src="https://raw.githubusercontent.com/maxhormazabal-test/nlu-p1/main/2.png" width="700">

- Click on one of them (could be anyone) and then click on "View raw"

<img src="https://raw.githubusercontent.com/maxhormazabal-test/nlu-p1/main/3.png" width="700">

- Here there is the raw file, this is the information that we need, our data. Just copy the URL.

<div style="border: 1px solid #2596be;padding:10px;">
    <p style="text-align: center;">**Important:** You do not need to repeat this process for thi others two files. We need only one URL for **each language**.</p>

</div>

<img src="https://raw.githubusercontent.com/maxhormazabal-test/nlu-p1/main/4.png" width="700">

### 3. Read the data with Python

Now we have the address of the data, let's read this file and save as a Python data structure. This is an example with _English Data_ using the URL `https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu` ins this case is the URL for training but as we wrote previously, does not matter wich data set you chose. We just need one of them.

Distribute the URL in two variables:

    - base_url: Contain the URL until "/master/"
    - file_basename: Contain the name of the language without the last score
    
It will be clear with the next example:

```python
# English
# 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu'

base_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/'
file_basename = 'en_ewt-ud'
```

After you did this with all the languages that you want we can start with the preprocessing.

<div style="border: 1px solid #2596be;padding:10px;">
    <p style="text-align: center;"> <strong> Important:</strong> We are going to use self-made functions to make this process more understandable, the finall cell contains all the functions to be runned together (We must have this functions in memory to make the notebook works.</p>

</div>

## 4. Preprocessing data

The function `preprocessingOneStep` contains all the preprocessing steps just in one function. The output are the data sets and the path of that data. Actually the notebook of preproceesing is only in charge of this part of the project and the data we will use in models will be saved as files and the mentioned relative path.

With the variables:

```
stack_len = 7
buffer_len = 10
```

You can set the size of stack and buffer for the sets. Run this cell everytime you need to create a new data source for models

The code is the following:

```python
base_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-ParTUT/master/'
file_basename = 'en_partut-ud'
stack_len = 7
buffer_len = 10

(path,
x_train_token,action_encod_train,deprel_encod_train,
x_test_token,action_encod_test,deprel_encod_test,
x_val_token,action_encod_val,deprel_encod_val) = preprocessingOneStep(base_url,file_basename,stack_len,buffer_len)


!mkdir -p {path}


saveData(path,x_train_token,action_encod_train,deprel_encod_train,x_test_token,action_encod_test,deprel_encod_test,x_val_token,action_encod_val,deprel_encod_val)
```

Notice that after the execution of `preprocessingOneStep` the line `!mkdir -p {path}` is runned because the `!` symbol let us work with the terminal prompt and create a directory to save the numpy files for each set.

After creating the folder we use `saveData` for save the files in your Google Drive directory. This is a fundamental step because the model notebook is going to take the data just reading the files; that way we do not have to repeat the preprocessing everytime.

**If you run many times the folder should be like this**

<img src="https://raw.githubusercontent.com/maxhormazabal/depencendy_parsing/main/img/nlu_data_folder.png" width="300">

## Models Notebook

After generating the data needed to train the models, the neural network tests begin. For this it is necessary to know the relevant functions:

1. `buildModelA` allows us to build and compile the model selecting all the important parameters. It is crucial to notice that `buildModelA` is just an example of how the arquitectures are working inside a function. If you want to know more about all the tested arquitectures in this project you can consult the optional file `arquitectures.md` that shows the arquitectures used one by one.

```python
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
```

2. `fitModel` use the model created with the previous function and train it.

```python
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

```

3. Finally `saveModelAData` is a function that can be used to test different configuration of an specific parameter. It trains a model changing the selected pivot parameter and save the results as a dataframe. The output of this function is the summarized data set and the best model of each execution.

**Important**: The metrics are calcutated with the evaluation function (Keras) using the testing set.

```python
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

```

One execution of those three function together is the following

```python
# Testing different optimizers

pivot_name = 'optimizer'
pivot = [
    ('adam' , 0.001),
    ('adam' , 0.01),
    ('adam' , 0.1),
    ('sgd' , 0.001),
    ('sgd' , 0.01),
    ('sgd' , 0.1),
    ('rmsprop' , 0.001),
    ('rmsprop' , 0.01),
    ('rmsprop' , 0.1),
    ('adadelta' , 0.001),
    ('adadelta' , 0.01),
    ('adadelta' , 0.1),
    ('adagrad' , 0.001),
    ('adagrad' , 0.01),
    ('adagrad' , 0.1),
    ('adamax' , 0.001),
    ('adamax' , 0.01),
    ('adamax' , 0.1),
    ('nadam' , 0.001),
    ('nadam' , 0.01),
    ('nadam' , 0.1),
    ('ftrl' , 0.001),
    ('ftrl' , 0.01),
    ('ftrl' , 0.1)
]

# Setting values
stack_len = 3
buffer_len = 3
arquitecture = 'A'
stopper = 'val_Deprel_Output_loss'
patience = 3
batch_size = 1024
epochs = 10
optimizer = 'adam'
learning_rate = 0.01
embedd_output = 50

(df_res_optimizers,models_optimizers) = saveModelAData(pivot_name,pivot,
                   x_train_stack,x_train_buffer,action_train, deprel_train,
                   x_val_stack,x_val_buffer,action_val,deprel_val,
                   x_test_stack,x_test_buffer,action_test,deprel_test,
                   stack_len,buffer_len,
                   stopper,patience,
                   batch_size,epochs,
                   optimizer,learning_rate,
                   embedd_output=50)
```

Where the data frame of result is like this

<img src="https://raw.githubusercontent.com/maxhormazabal/depencendy_parsing/main/img/df_results_example.png" width="700">

And you can find all the data set with the results (based on a pivot parameter) in the folder `model_testing` in `.csv` format.