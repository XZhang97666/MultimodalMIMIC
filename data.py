import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import random
import pickle
from torch.nn.utils.rnn import pad_sequence
# import argparse
import os
from transformers import AutoTokenizer, BertTokenizer
import torch.nn as nn
from until import *



def data_perpare(args,mode,tokenizer,data=None):
    dataset=TSNote_Irg(args,mode, tokenizer,data=data)
    if mode=='train':
        sampler = RandomSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,collate_fn=TextTSIrgcollate_fn)
    else:
        sampler = SequentialSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,collate_fn=TextTSIrgcollate_fn)


    return dataset, sampler, dataloader


def F_impute(X,tt,mask,duration,tt_max):
    no_feature=X.shape[1]
    impute=np.zeros(shape=(tt_max//duration,no_feature*2))
    for  x,t,m in zip(X,tt,mask):
        row=int(t/duration)
        if row>=tt_max:
            continue
        for  f_idx, (rwo_x, row_m) in enumerate(zip(x,m)):
            if row_m==1:
                impute[row][no_feature+f_idx]=1
                impute[row][f_idx]=rwo_x
            else:
                if impute[row-1][f_idx]!=0:
                    impute[row][f_idx]=impute[row-1][f_idx]


    return impute



class TSNote_Irg(Dataset):

    def __init__(self,args,mode,tokenizer,chunk=False,data=None):
        self.tokenizer = tokenizer
        self.max_len = args.max_length
        if data !=None:
            self.data=data
        else:
            self.data =load_data(file_path=args.file_path,mode=mode,debug=args.debug)
        self.chunk=args.chunk
        if self.chunk:
            self.text_id_attn_data = load_data(file_path=args.file_path,mode=mode,text=True)
        self.padding= "max_length" if args.pad_to_max_length  else False

        if mode=="train":
            self.notes_order=args.notes_order
        else:
            self.notes_order="Last"

        if args.ratio_notes_order!=None:
            self.order_sample=np.random.binomial(1, args.ratio_notes_order,len(self.data))

        self.modeltype=args.modeltype
        self.model_name=args.model_name
        self.num_of_notes=args.num_of_notes
        self.tt_max=args.tt_max
    def __getitem__(self, idx):

        if self.notes_order!=None:

            notes_order=self.notes_order
        else:
            notes_order= 'Last' if self.order_sample[idx]==1  else 'First'
        data_detail = self.data[idx]
        idx=data_detail['data_names']
        reg_ts=data_detail['TS_data']
        ts=data_detail['irg_ts']


        ts_mask=data_detail['irg_ts_mask']
        text = data_detail['text_data']



        if len(text)==0:
            return None
        text_token=[]
        atten_mask=[]
        label=data_detail["label"]
        ts_tt=data_detail["ts_tt"]
        start_time=data_detail["adm_time"]
        text_time_to_end=data_detail["text_time_to_end"]

        reg_ts=F_impute(ts,ts_tt,ts_mask,1,self.tt_max)
        if 'Text' in self.modeltype :
            for t in text:
                inputs = self.tokenizer.encode_plus(t, padding=self.padding,\
                                                    max_length=self.max_len,\
                                                    add_special_tokens=True,\
                                                    return_attention_mask = True,\
                                                    truncation=True)
                text_token.append(torch.tensor(inputs['input_ids'],dtype=torch.long))
                attention_mask=inputs['attention_mask']
                if "Longformer" in self.model_name :

                    attention_mask[0]+=1
                    atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))
                else:
                    atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))


        label=torch.tensor(label,dtype=torch.long)
        reg_ts=torch.tensor(reg_ts,dtype=torch.float)
        ts=torch.tensor(ts,dtype=torch.float)
        ts_mask=torch.tensor(ts_mask,dtype=torch.long)
        ts_tt=torch.tensor([t/self.tt_max for t in ts_tt],dtype=torch.float)
        text_time_to_end=[1-t/self.tt_max for t in text_time_to_end]
        text_time_mask=[1]*len(text_time_to_end)

        if 'Text' in self.modeltype :
            while len(text_token)<self.num_of_notes:
                text_token.append(torch.tensor([0],dtype=torch.long))
                atten_mask.append(torch.tensor([0],dtype=torch.long))
                text_time_to_end.append(0)
                text_time_mask.append(0)


        text_time_to_end=torch.tensor(text_time_to_end,dtype=torch.float)
        text_time_mask=torch.tensor(text_time_mask,dtype=torch.long)

        if 'Text' not in self.modeltype:
            return {'idx':idx,'ts':ts, 'ts_mask': ts_mask, 'ts_tt': ts_tt, 'reg_ts':reg_ts,"label":label}
        if notes_order=="Last":
            return {'idx':idx,'ts':ts, 'ts_mask': ts_mask, 'ts_tt': ts_tt,'reg_ts':reg_ts, "input_ids":text_token[-self.num_of_notes:],"label":label, "attention_mask":atten_mask[-self.num_of_notes:], \
            'note_time':text_time_to_end[-self.num_of_notes:], 'text_time_mask': text_time_mask[-self.num_of_notes:],
               }
        else:
            return {'idx':idx,'ts':ts, 'ts_mask': ts_mask, 'ts_tt': ts_tt, 'reg_ts':reg_ts,"input_ids":text_token[:self.num_of_notes],"label":label, "attention_mask":atten_mask[:self.num_of_notes] ,\
             'note_time':text_time_to_end[:self.num_of_notes],'text_time_mask': text_time_mask[:self.num_of_notes]
               }

    def __len__(self):
        return len(self.data)



def load_data(file_path,mode,debug=False,text=False):
    if not text:
        dataPath = os.path.join(file_path, mode + 'p2x_data.pkl')
    else:
        dataPath = os.path.join(file_path, mode + 'token_attn.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if debug and not text:
                data=data[:100]

    return data


def TextTSIrgcollate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: len(x['ts']) <1000, batch))
    ts_input_sequences=pad_sequence([example['ts'] for example in batch],batch_first=True,padding_value=0 )
    ts_mask_sequences=pad_sequence([example['ts_mask'] for example in batch],batch_first=True,padding_value=0 )
    ts_tt=pad_sequence([example['ts_tt'] for example in batch],batch_first=True,padding_value=0 )
    label=torch.stack([example["label"] for example in batch])

    reg_ts_input=torch.stack([example['reg_ts'] for example in batch])
    if len(batch[0])>6:
        input_ids=[pad_sequence(example['input_ids'],batch_first=True,padding_value=0).transpose(0,1) for example in batch]
        attn_mask=[pad_sequence(example['attention_mask'],batch_first=True,padding_value=0).transpose(0,1) for example in batch]

        input_ids=pad_sequence(input_ids,batch_first=True,padding_value=0).transpose(1,2)
        attn_mask=pad_sequence(attn_mask,batch_first=True,padding_value=0).transpose(1,2)

        note_time=pad_sequence([torch.tensor(example['note_time'],dtype=torch.float) for example in batch],batch_first=True,padding_value=0)
        note_time_mask=pad_sequence([torch.tensor(example['text_time_mask'],dtype=torch.long) for example in batch],batch_first=True,padding_value=0)

    else:
        input_ids,attn_mask, note_time, note_time_mask =None,None,None,None
    return ts_input_sequences,ts_mask_sequences, ts_tt, reg_ts_input, \
         input_ids,attn_mask, note_time ,note_time_mask, label





