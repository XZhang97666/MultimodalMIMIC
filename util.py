import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../TS/mimic3-benchmarks')
sys.path.insert(0, '../ClinicalNotes_TimeSeries/models')
import pickle
import re
import numpy as np
import json
from data import *
import statistics as stat
logger = None
import  argparse
import pickle
from accelerate import Accelerator
from sklearn import metrics

from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          AdamW,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel,
                          LongformerConfig,
                          LongformerModel,
                          LongformerTokenizer,

                         )

def parse_args():
    parser = argparse.ArgumentParser(description="Alignment text and ts data")
    parser.add_argument(
            "--task", type=str, default="ihm"
        )
    parser.add_argument(
        "--file_path", type=str, default="Data", help="A path to dataset folder"
    )
    parser.add_argument("--output_dir", type=str, default="Checkpoints", help="Where to store the final model.")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--modeltype", type=str, default="TS_Text", help="TS, Text or TS_Text")
    parser.add_argument("--eval_score", default=['auc', 'auprc', 'f1'], type=list)

    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated," " sequences shorter will be padded if `--pad_to_max_lengh` is passed."),)
    parser.add_argument( "--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument( "--model_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size  for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument("--num_update_bert_epochs", type=int, default=10, help="Number of per training epochs update the bert model.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--txt_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for Txt self-attention and Bert to use.",
    )

    parser.add_argument(
        "--ts_learning_rate",
        type=float,
        default=0.0004,
        help="Initial learning rate for TS self-attention to use.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument( "--pt_mask_ratio",default=0.15, type=float, help="mask rate for pretrain .",
    )
    parser.add_argument( "--mean_mask_length",default=3, type=int, help="mean mask length for pretrain .",
    )

    parser.add_argument('--chunk', action='store_true')
    parser.add_argument("--chunk_type", default='sent_doc_pos', type=str, help="How to chunk the text. sent_doc_pos: sentence level position + doc level position")
    parser.add_argument("--warmup_proportion", default=0.10, type=float, help="proportion for the warmup in the lr scheduler.")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for CNN.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer encoder layer.")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of transformer cross encoder layer.")
    parser.add_argument("--embed_dim", default=30, type=int, help="attention embedding dim.")

    parser.add_argument("--irregular_learn_emb_ts", action='store_true')
    parser.add_argument("--irregular_learn_emb_text", action='store_true')
    parser.add_argument("--reg_ts", action='store_true')
    parser.add_argument("--tt_max", default=48, type=int, help="max time for irregular time series.")
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument('--ts_to_txt', action='store_true')
    parser.add_argument('--txt_to_ts', action='store_true')

    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument("--model_name", default='BioBert', type=str, help="model for text")
    parser.add_argument('--num_of_notes', help='Number of notes to include for a patient input 0 for all the notes', type=int, default=5)
    parser.add_argument('--notes_order', help='Should we get notes from beginning of the admission time or from end of it, options are: 1. First: pick first notes 2. Last: pick last notes', default=None)
    parser.add_argument('--ratio_notes_order', help='The parameter of a bernulli distribution on whether take notes from First or Last, 1-Last, 0-First',type=float, default=None)

    parser.add_argument('--bertcount',type=int, default=3,help='number of count update bert in total')
    parser.add_argument('--first_n_item', help='Top n item in val seeds', type=int, default=3)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--self_cross', action='store_true')
    parser.add_argument('--TS_mixup', action='store_true', help='mix up reg and irg data')
    parser.add_argument("--mixup_level", default=None, type=str, help="mixedup level for two time series data, choose: 'batch', batch_seq' or 'batch_seq_feature'. ")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--FTLSTM', action='store_true')
    parser.add_argument('--Interp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--datagereate_seed", type=int, default=42, help="A seed for reproducible data generation .")
    parser.add_argument("--TS_model", type=str, default='Atten', help="LSTM, CNN, Atten")
    parser.add_argument("--cross_method", default='self_cross', type=str, help="baseline fusion method: MAGGate, MulT, Outer,concat")
    args = parser.parse_args()

    return args

def loadBert(args,device):
    if args.model_name!=None:
        if args.model_name== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            BioBert=AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.model_name=="bioRoberta":
            config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        elif  args.model_name== "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")
        elif args.model_name== "bioLongformer":
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            BioBert= AutoModel.from_pretrained("yikuan8/Clinical-Longformer")

        else:
            raise ValueError("model_name should be BioBert,bioRoberta,bioLongformer or Bert")
    else:
        if args.model_path!=None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            BioBert = AutoModel.from_pretrained(args.model_path)
        else:
            raise ValueError("provide either model_name or model_path")

    BioBert = BioBert.to(device)
    BioBertConfig = BioBert.config
    return BioBert,  BioBertConfig,tokenizer





def data_generate(args):
    dataPath = os.path.join(args.file_path,  'all_data_p2x_data.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if args.debug:
                data=data[:100]

    data=np.array(data)
    total_num=len(data)
    idx=np.arange(total_num)

    np.random.seed(args.seed)
    np.random.shuffle(idx)

    train= data[idx[:int(len(idx)*0.8)]]
    print(train[0]['data_names'])
    val=data[idx[int(len(idx)*0.8):int(len(idx)*0.9)]]
    test=data[idx[int(len(idx)*0.9):]]

    train=train.tolist()
    val=val.tolist()
    test=test.tolist()
    return train, val, test








def metrics_multilabel(y_true, predictions, verbose=1):
    # import pdb; pdb.set_trace()
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return{"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


def diff_float(time1, time2):
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0

def get_time_to_end_diffs(times, starttimes):

    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')
        for t in times:
            time = np.datetime64(t)
            dt = diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def change_data_form(file_path,mode,debug=False):
    dataPath = os.path.join(file_path, mode + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            # (data, _, _, _) = pickle.load(f)
            data = pickle.load(f)
            if debug:
                data=data[:500]

        data_X = data[0]
        data_y = data[1]
        data_text = data[2]
        data_names = data[3]
        start_times = data[4]
        timetoends = data[5]

        dataList=[]

        assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 


        assert  len(data_text[0])==len(timetoends[0])
        for x,y, text, name, start, end in zip(data_X,data_y,data_text, data_names,start_times,timetoends):
            if len(text)==0:
                continue
            new_text=[]
            for t in text:
                # import pdb;
                # pdb.set_trace()
                t=re.sub(r'\s([,;?.!:%"](?:\s|$))', r'\1', t)
                t=re.sub(r"\b\s+'\b", r"'", t)
                new_text.append(t.lower().strip())


            data_detail={"data_names":name,
                         "TS_data":x,
                         "text_data":new_text,
                        "label":y,
                         "adm_time":start,
                         "text_time_to_end":end
                        }
            dataList.append(data_detail)

    os.makedirs('Data',exist_ok=True)
    dataPath2 = os.path.join(file_path, mode + 'p2x_data.pkl')

    with open(dataPath2, 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump(dataList, f)

    return dataList

def data_replace(file_path1,file_path2,mode,debug=False):
    dataPath1 = os.path.join(file_path2, mode + '.pkl')
    dataPath2 = os.path.join(file_path1, mode + 'p2x_data.pkl')
    if os.path.isfile(dataPath1):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath1)
        with open(dataPath1, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data=data[:500]

    with open(dataPath2, 'rb') as f:
            data_r=pickle.load(f)
    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    data_names = data[3]
    start_times = data[4]
    timetoends = data[5]
    data_dict={}

    assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 



    assert  len(data_text[0])==len(timetoends[0])
    for x,name in zip(data_X, data_names):

        data_dict[name]=x
    for idx, data_detail in enumerate(data_r):
        new_x=data_dict[data_detail['data_names']]
        data_detail['TS_data']=new_x


    dataPath3=os.path.join(file_path2, mode + 'p2x_data.pkl')
    with open(dataPath3, 'wb') as f:
        pickle.dump(data_r, f)





def merge_reg_irg(dataPath_reg, dataPath_irg):
    with open(dataPath_irg, 'rb') as f:
        data_irg=pickle.load(f)

    with open(dataPath_reg, 'rb') as f:
        data_reg=pickle.load(f)


    for idx, data_dict in enumerate(data_reg):
        irg_dict=data_irg[data_dict['data_names']]
        data_dict['ts_tt']=irg_dict['ts_tt']
        data_dict['irg_ts']=irg_dict['irg_ts']
        data_dict['irg_ts_mask']=irg_dict['irg_ts_mask']

        assert (data_dict['label']==irg_dict['label']).all()

    with open(dataPath_reg, 'wb') as f:
        pickle.dump(data_reg,f)






