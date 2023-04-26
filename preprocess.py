

import numpy as np
import os
import pickle
import argparse

import json
import statistics as stat
import sys
from pathlib import Path
sys.path.append('../mimic3-benchmarks')

from mimic3benchmark.readers import InHospitalMortalityReader,PhenotypingReader
from mimic3models import common_utils
from text_utils import *


def diff_float(time1, time2):
    # compute time2-time1
    # return differences in hours but as float
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0


def get_time_to_end_diffs(times, st,endtime=49):
    timetoends = []
    difftimes = []
    et = np.datetime64(st) + np.timedelta64(endtime, 'h')
    for t in times:
        time = np.datetime64(t)
        dt = diff_float(time, et)
        assert dt >= 0 #delta t should be positive
        difftimes.append(dt)
    timetoends.append(difftimes)
    return timetoends


def extract_irregular(dataPath_in,dataPath_out):
    # Opening JSON file
    channel_info_file = open('Data/irregular/channel_info.json')
    dis_config_file=open('Data/irregular/discretizer_config.json')
    channel_info = json.load(channel_info_file)
    dis_config=json.load(dis_config_file)
    channel_name=dis_config['id_to_channel']
    is_catg=dis_config['is_categorical_channel']


    with open(dataPath_in, 'rb') as f:
        X,y,names = pickle.load(f)
    data_irregular=[]

    for p_id, x, in enumerate(X):
        data_i={}
        tt=[]
        features_list=[]
        features_mask_list=[]
        for t_idx, feature in enumerate(x):
            f_list_per_t=[]
            f_mask_per_t=[]
            for f_idx, val in enumerate(feature):
                if f_idx==0:
                    tt.append(round(float(val),2))
                else:
                    head=channel_name[f_idx-1]
                    if val=='':
                        f_list_per_t.append(0)
                        f_mask_per_t.append(0)
                    else:
                        f_mask_per_t.append(1)
                        if is_catg[head]:
                            val=channel_info[head]['values'][val]
                        f_list_per_t.append(float(round(float(val),2)))
            assert len(f_list_per_t)==len(f_mask_per_t)
            features_list.append(f_list_per_t)
            features_mask_list.append(f_mask_per_t)
        assert len(features_list)==len(features_mask_list)==len(tt)
        data_i['name']=names[p_id]
        data_i['label']=y[p_id]
        data_i['ts_tt']=tt
        data_i['irg_ts']=np.array(features_list)
        data_i['irg_ts_mask']=np.array(features_mask_list)
        data_irregular.append(data_i)
    with open(dataPath_out, 'wb') as f:
        pickle.dump(data_irregular, f)

    channel_info_file.close()
    dis_config_file.close()


def mean_std(dataPath_in,dataPath_out):
    with open(dataPath_in, 'rb') as f:
        data=pickle.load(f)
    feature_list=[[] for _ in range(17)]
    for p_id, p_data in enumerate(data):
        irg_ts=p_data['irg_ts']
        irg_ts_mask=p_data['irg_ts_mask']

        for t_idx, (ts, ts_mask) in enumerate(zip(irg_ts,irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, ts_mask)):
                # print(f_idx)
                if mask_val==1:
                    feature_list[f_idx].append(val)


    means=[]
    stds=[]

    for f_vals in feature_list:
        means.append(stat.mean(f_vals))
        stds.append(stat.stdev(f_vals))

    with open(dataPath_out, 'wb') as f:
        pickle.dump((means,stds), f)



def normalize(dataPath_in,dataPath_out,normalizer_path):
    with open(dataPath_in, 'rb') as f:
        data=pickle.load(f)

    with open(normalizer_path, 'rb') as f:
        means, stds=pickle.load(f)

    for p_id, p_data in enumerate(data):
        irg_ts=p_data['irg_ts']
        irg_ts_mask=p_data['irg_ts_mask']

        for t_idx, (ts, ts_mask) in enumerate(zip(irg_ts,irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, ts_mask)):
                if mask_val==1:
                    irg_ts[t_idx][f_idx]=(val-means[f_idx])/stds[f_idx]
    with open(dataPath_out, 'wb') as f:
        pickle.dump(data,f)



def save_data(reader,outputdir,small_part=False,mode=None):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]

    os.makedirs(outputdir,exist_ok=True)
    
    with open(outputdir+"irregular_"+mode+".pkl", 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump((data,labels,names), f)
    
    return 
        


def merge_text_ts(textdict, timedict, start_times,tslist,period_length,dataPath_out):
    suceed = 0
    missing = 0
    for idx, ts_dict in enumerate(tslist):
        name=ts_dict['name']
        if name in textdict:
            ts_dict['text_data']=textdict[name]
            # ts_dict['text_times']=timedict[name]
            # ts_dict['text_start_times']=start_times[name]
            ts_dict['text_time_to_end']= get_time_to_end_diffs(timedict[name], start_times[name],endtime=period_length+1)
            suceed += 1
        else:
            missing += 1

    print("Suceed Merging: ", suceed)
    print("Missing Merging: ", missing)

    with open(dataPath_out, 'wb') as f:
        pickle.dump(tslist, f)
    return 




if __name__ == "__main__":
    dir_input="../mimic3-benchmarks/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the data of task',
                        default=os.path.join(os.path.dirname(__file__), '../mimic3-benchmarks/data/in-hospital-mortality/')) #'../mimic3-benchmarks/data/phenotyping/'
    parser.add_argument('--small_part', action='store_true')
    parser.add_argument("--period_length", default=48, type=int, help="period length of reader.") #24
    parser.add_argument("--task", default='ihm', type=str, help="task name to create data")
    parser.add_argument("--outputdir", default='./Data/', type=str, help="data output dir") #'./Data/pheno/'
    args = parser.parse_args()
    print(args)

    output_dir=args.outputdir+args.task+"/"

    if args.task=='ihm':


        # Build readers, discretizers, normalizers
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                                listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                period_length=args.period_length)
        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=args.period_length)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=args.period_length)


    elif args.task=='pheno':
        
            # Build readers, discretizers, normalizers
        train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                                listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                period_length=args.period_length)
        val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=args.period_length)

        test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=args.period_length)

    save_data(train_reader, output_dir,args.small_part,mode='train')
    save_data(val_reader, output_dir,args.small_part,mode='val')
    save_data(test_reader,output_dir,args.small_part,mode='test')


    for mode in ['train', 'val', 'test']:
        extract_irregular(output_dir+'irregular_'+mode+'.pkl',output_dir+'irregular_'+mode+'.pkl')
    #calculate mean,std of ts
    mean_std(output_dir+'irregular_train.pkl',args.outputdir+'irregular/mean_std.pkl')

    for mode in ['train', 'val', 'test']:
        normalize(output_dir+'irregular_'+mode+'.pkl',\
                  output_dir+'norm_irregular_'+mode+'.pkl',\
                  args.outputdir+'irregular/mean_std.pkl')

    textdata_fixed = "../mimic3-benchmarks/data/root/text_fixed/"
    starttime_path = "../mimic3-benchmarks/starttime.pkl"
    test_textdata_fixed = "../mimic3-benchmarks/data/root/test_text_fixed/"
    test_starttime_path = "../mimic3-benchmarks/test_starttime.pkl"

    
    for mode in ['train', 'val', 'test']:
        with open(output_dir+'norm_irregular_'+mode+'.pkl', 'rb') as f:
            tsdata=pickle.load(f)
        
        names = [data['name'] for data in tsdata]

        if (mode == 'train') or (mode == 'val'):
            text_reader = TextReader(textdata_fixed, starttime_path)
        else:
            text_reader = TextReader(test_textdata_fixed, test_starttime_path)
        
        data_text, data_times, data_time = text_reader.read_all_text_append_json(names, args.period_length)
        merge_text_ts(data_text, data_times, data_time,tsdata, args.period_length,output_dir+mode+'p2x_data.pkl')
        


        

        
        