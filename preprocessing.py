

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
from mimic3models.preprocessing import Normalizer
from text_utils import *





class Discretizer_multi:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), 'Data/irregular/discretizer_config.json'),channel_path=os.path.join(os.path.dirname(__file__), 'Data/irregular/channel_info.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']
            
        with open(channel_path) as f:
            self.channel_info = json.load(f)

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)


        data = np.zeros(shape=(N_bins, N_channels), dtype=float)
        # data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value):
            channel_id = self._channel_to_id[channel]
         
            if self._is_categorical_channel[channel]:

                value=self.channel_info[channel]['values'][value]
            
            data[bin_id,channel_id] = float(value)
                
        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j])
                original_value[bin_id][channel_id] = row[j]

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    # write(data, bin_id, channel, imputed_value, begin_pos)
                    write(data, bin_id, channel, imputed_value)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    # write(data, bin_id, channel, imputed_value, begin_pos)
                    write(data, bin_id, channel, imputed_value)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []

        for channel in self._id_to_channel:
            new_header.append(channel)
        for channel in self._id_to_channel:

            new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)
    

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
        ireg_data,reg_data,y,names = pickle.load(f)
    data_irregular=[]

    for p_id, x, in enumerate(ireg_data):
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
        data_i['reg_ts']=reg_data[p_id]
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
    
    irg_f_num=data[0]['irg_ts'].shape[1]
    reg_f_num=data[0]['reg_ts'].shape[1]
    irg_feature_list=[[] for _ in range(irg_f_num)]
    reg_feature_list=[[] for _ in range(reg_f_num)]
    for p_id, p_data in enumerate(data):
        irg_ts=p_data['irg_ts']
        irg_ts_mask=p_data['irg_ts_mask']
        reg_ts=p_data['reg_ts']

        for t_idx, (ts, mask) in enumerate(zip(irg_ts,irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, mask)):
                # print(f_idx)
                if mask_val==1:
                    irg_feature_list[f_idx].append(val)
        
        for ts in reg_ts:
            for f_idx, (val, mask_val) in enumerate(zip(ts[:reg_f_num//2], ts[reg_f_num//2:])):
                reg_feature_list[f_idx].append(val)
        



    irg_means=[]
    irg_stds=[]
    reg_means=[]
    reg_stds=[]

    for irg_vals,reg_vals in zip(irg_feature_list,reg_feature_list):
        irg_means.append(stat.mean(irg_vals))
        irg_stds.append(stat.stdev(irg_vals))
        reg_means.append(stat.mean(reg_vals))
        reg_stds.append(stat.stdev(reg_vals))
    with open(dataPath_out, 'wb') as f:
        pickle.dump((irg_means,irg_stds,reg_means,reg_stds), f)



def normalize(dataPath_in,dataPath_out,normalizer_path):

    
    with open(dataPath_in, 'rb') as f:
        data=pickle.load(f)

    with open(normalizer_path, 'rb') as f:
        irg_means,irg_stds,reg_means,reg_stds=pickle.load(f)

    for p_id, p_data in enumerate(data):
        irg_ts=p_data['irg_ts']
        irg_ts_mask=p_data['irg_ts_mask']

        reg_ts=p_data['reg_ts']
        feature_dim=irg_ts.shape[1]

        for t_idx, (ts, ts_mask) in enumerate(zip(irg_ts,irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, ts_mask)):
                if mask_val==1:
                    irg_ts[t_idx][f_idx]=(val-irg_means[f_idx])/irg_stds[f_idx]
        
        for t_idx, ts in enumerate(reg_ts):
            for f_idx, val in enumerate(ts[:feature_dim]):
                reg_ts[t_idx][f_idx]=(val-reg_means[f_idx])/reg_stds[f_idx]

    # import pdb;pdb.set_trace()
    with open(dataPath_out, 'wb') as f:
        pickle.dump(data,f)



def save_data(reader,discretizer, outputdir,small_part=False,mode=None,non_mask=None):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    irg_data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    reg_data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(irg_data, ts)]
    os.makedirs(outputdir,exist_ok=True)
    with open(outputdir+"ts_"+mode+".pkl", 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump((irg_data,reg_data, labels,names), f)
    
    return 



def merge_text_ts(textdict, timedict, start_times,tslist,period_length,dataPath_out):
    suceed = 0
    missing = 0
    for idx, ts_dict in enumerate(tslist):
        name=ts_dict['name']
        if name in textdict:
            ts_dict['text_data']=textdict[name]
   
            ts_dict['text_time_to_end']= get_time_to_end_diffs(timedict[name], start_times[name],endtime=period_length+1)[0]
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
    parser.add_argument("--period_length", default=48, type=int, help="period length of reader.") #24
    parser.add_argument("--task", default='ihm', type=str, help="task name to create data")
    parser.add_argument("--outputdir", default='./Data/', type=str, help="data output dir") #'./Data/pheno/'
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
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


    discretizer = Discretizer_multi(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    
    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = '../mimic3-benchmarks/mimic3models/in_hospital_mortality/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    save_data(train_reader, discretizer, output_dir,args.small_part,mode='train')
    save_data(val_reader, discretizer, output_dir,args.small_part,mode='val')
    save_data(test_reader, discretizer,  output_dir,args.small_part,mode='test')




    for mode in ['train', 'val', 'test']:
        extract_irregular(output_dir+'ts_'+mode+'.pkl',output_dir+'ts_'+mode+'.pkl')
    #calculate mean,std of ts
    mean_std(output_dir+'ts_train.pkl', output_dir+'mean_std.pkl')

    for mode in ['train', 'val', 'test']:
        normalize(output_dir+'ts_'+mode+'.pkl',\
                  output_dir+'norm_ts_'+mode+'.pkl',\
                   output_dir+'mean_std.pkl')

    textdata_fixed = "../mimic3-benchmarks/data/root/text_fixed/"
    starttime_path = "../mimic3-benchmarks/starttime.pkl"
    test_textdata_fixed = "../mimic3-benchmarks/data/root/test_text_fixed/"
    test_starttime_path = "../mimic3-benchmarks/test_starttime.pkl"

    
    for mode in ['train', 'val', 'test']:
        with open(output_dir+'norm_ts_'+mode+'.pkl', 'rb') as f:
            tsdata=pickle.load(f)
        
        names = [data['name'] for data in tsdata]

        if (mode == 'train') or (mode == 'val'):
            text_reader = TextReader(textdata_fixed, starttime_path)
        else:
            text_reader = TextReader(test_textdata_fixed, test_starttime_path)
        
        data_text, data_times, data_time = text_reader.read_all_text_append_json(names, args.period_length)
        merge_text_ts(data_text, data_times, data_time,tsdata, args.period_length,output_dir+mode+'p2x_data.pkl')
        


        

        
        