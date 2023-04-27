import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import warnings
import time
import logging
logger = logging.getLogger(__name__)
from model import *
from train import *
from checkpoint import *
from util import *
from accelerate import Accelerator
from interp import *




def main():
    args = parse_args()
    print(args)

    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"
    accelerator = Accelerator(fp16=args.fp16, mixed_precision=args.mixed_precision,cpu=args.cpu)

    device = accelerator.device
    print(device)
    os.makedirs(args.output_dir, exist_ok = True)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)


    make_save_dir(args)

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    if args.mode=='train':
        if 'Text' in args.modeltype:
            BioBert, BioBertConfig,tokenizer=loadBert(args,device)
        else:
            BioBert,tokenizer=None,None
        train_dataset, train_sampler, train_dataloader=data_perpare(args,'train',tokenizer)
        val_dataset, val_sampler, val_dataloader=data_perpare(args,'val',tokenizer)
        _, _, test_data_loader=data_perpare(args,'test',tokenizer)


    if 'Text' in args.modeltype:
        model= MULTCrossModel(args=args,device=device,orig_d_ts=17, orig_reg_d_ts=34, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert)
    else:
        model= TSMixed(args=args,device=device,orig_d_ts=17,orig_reg_d_ts=34, ts_seq_num=args.tt_max)

    print(device)



    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='Text' or args.modeltype=='TS_Text':
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params':[p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")


    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader,val_dataloader,test_data_loader)


    trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer)
    eval_test(args,model,test_data_loader, device)



if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
