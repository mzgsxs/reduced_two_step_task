import numpy as np
import os, re, pickle
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def load_all_data(extract_type='Training', batch_name = 'batch1'):
  #save the file
  data_pickle_dir = os.path.join(data_dir, 'pickle', batch_name, extract_type)
  f = open(os.path.join(data_pickle_dir, 'file.pickle'),'rb')
  data = pickle.load(f)
  f.close()
  return data

 
if __name__ == "__main__":
  data = load_all_data()
  subjects = list(data.keys())
  subject_name = subjects[0]
  print(data[subject_name][0])



