import pandas as pd
import numpy as np

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