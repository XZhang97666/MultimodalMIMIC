

import numpy as np
import os
import pickle
import argparse


import sys
from pathlib import Path
sys.path.append('../mimic3-benchmarks')

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../mimic3-benchmarks/data/in-hospital-mortality/'))
parser.add_argument('--small_part', action='store_true')
args = parser.parse_args()
print(args)



def save_data(reader, small_part=False,mode=None):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    
    with open("irregular_"+mode+".pkl", 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump((data,labels,names), f)
    
    return
        




# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)
val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)


save_data(train_reader, args.small_part,mode='train')
save_data(val_reader, args.small_part,mode='val')
save_data(test_reader,args.small_part,mode='test')



