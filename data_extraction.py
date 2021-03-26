import numpy as np
import os, ast, re, pickle
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

get_stats = lambda x: re.search(r'P (?P<time>.*) T#:(?P<trail_num>.*) R#:(?P<num_reward>.*) B#:(?P<num_blocks>.*) C:(?P<action>.*) S:(?P<second_step_state>.*) O:(?P<reward>.*) CA:(?P<average_correct_rate>.*) B:(?P<block_type>.*) CT:(?P<choice_type>.*) TS:(?P<training_stage>.*)\n', x) 

batch_name = 'batch1'
extract_type = 'Experiment'




def extract_all_data():
  data_all = {}
  data_raw_dir = os.path.join(data_dir, batch_name)
  subjects = [ p for p in os.listdir(data_raw_dir) if os.path.isdir(os.path.join(data_raw_dir, p))]
  for subject_name in sorted(subjects):
    # data extraction
    subject_dir = os.path.join(data_raw_dir, subject_name, extract_type)
    data_subject = []
    for f_name in sorted(os.listdir(subject_dir)):
      f_path = os.path.join(subject_dir,f_name)
      print("processing file {}".format(f_path))
      data_session = data_extraction_session(f_path)
      data_subject.append(data_session)
    data_all[subject_name] = data_subject
  #save the file
  data_pickle_dir = os.path.join(data_dir, 'pickle', batch_name, extract_type)
  Path(data_pickle_dir).mkdir(parents=True, exist_ok=True)
  save_f = open(os.path.join(data_pickle_dir, 'file.pickle'),'wb')
  pickle.dump(data_all, save_f)
  save_f.close()
  


def data_extraction_session(f_path):
  # prepare storage instance
  data = {}
  data['state_transition'] = []
  data['choice_type'] = []
  data['block_type'] = []

  # reading
  f = open(f_path,'r')
  for line in f:
    if line[0] is "S":
      state_to_num_map = ast.literal_eval(line[2:-1])
    if line[0] is "E":
      event_to_num_map = ast.literal_eval(line[2:-1])
    if line[0] is "V":
      variable = re.search(r'V 0 (.*) (.*)\n', line)
      if variable is not None:
        data[variable.group(1)] = variable.group(2) 
    if line[0] is "P":
      trail = get_stats(line)
      if trail is not None:
        action = 1-int(trail.group('action'))
        state = 2-int(trail.group('second_step_state'))
        reward = float(trail.group('reward'))
        s1 = (0, action, 0., state)
        s2 = (state, 2, reward, -1)
        # ugly fix
        if data['training_stage'] in ['1.1', '1.2']:
            dat = {0:s2}
        else:
            dat = {0:s1, 1:s2}
        data['state_transition'].append(dat)
        data['choice_type'].append(trail.group('choice_type'))
        data['block_type'].append(trail.group('block_type'))
        #print(line)
        #print(dat)
  f.close()
  return data



if __name__ == "__main__":
  extract_all_data()
