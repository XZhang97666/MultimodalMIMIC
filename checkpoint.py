import re
import os
import torch
import operator
from statistics import mean,stdev
import fnmatch

import shutil
def copy_file(dst, src=os.getcwd()):

    pattern = "*.py"
    copy_dirs = [src,src+"/model"]
    pair_file_list = []
    for path, subdirs, files in os.walk(src):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                source_file = os.path.join(path, name)
                target_file = os.path.join(path, name).replace(src,dst)
                pair_file_list.append((source_file,target_file))


    for source_file,target_file in pair_file_list:
        if(os.path.dirname(source_file) in copy_dirs):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy(source_file, target_file)

def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
#         print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def make_save_dir(args):

    output_dir=args.output_dir+"/"+args.task+"/"+args.modeltype+"/"

    if 'Bert' in args.modeltype:
        output_dir+=args.model_name+"/"+args.notes_order+"/"+str(args.num_of_notes)+"/"
        output_dir+=str(args.max_length)+"_" +str(args.txt_learning_rate)+"_"+ str(args.num_train_epochs)+\
        "_"+str(args.embed_dim)+"_"+str(args.train_batch_size)+"_"+str(args.num_update_bert_epochs)+'/'

    else:

        if args.irregular_learn_emb_ts and "TS" in args.modeltype:
            output_dir+=  "TS_"+str(args.tt_max)+"/"+args.TS_model+"/"
        if args.irregular_learn_emb_text and 'Text' in args.modeltype:
            output_dir+= "Text_"+str(args.tt_max)+"/"+args.model_name+"/"+str(args.max_length)+"/"

        if args.modeltype=="TS_Text":
            if args.self_cross:
                if args.cross_method=="self_cross":
                    output_dir+='cross_attn'+str(args.cross_layers)+"/"
                else:
                    output_dir+=args.cross_method+"/"
            else:
                output_dir+=args.agg_type+"/"

        if args.modeltype=="Text" or args.modeltype=="TS":
            output_dir+='layer'+str(args.layers)+"/"

        if  args.TS_mixup:
            output_dir+=args.mixup_level+"/"

        if args.irregular_learn_emb_ts:
            output_dir+="irregular_TS_"+str(args.embed_time)+"/"
        else:
            output_dir+="regular_TS/"

        if args.irregular_learn_emb_text:
            output_dir+="irregular_Text_"+str(args.embed_time)+"/"
        else:
            output_dir+="regular_Text/"
        if "Text" in args.modeltype:
            output_dir+=str(args.txt_learning_rate)+"_"+str(args.num_update_bert_epochs)+"_"+str(args.bertcount)+"_"
        if "TS" in args.modeltype:
            output_dir+=str(args.ts_learning_rate)+"_"


        output_dir+= str(args.num_train_epochs)+"_"+str(args.num_heads)+"_"+str(args.embed_dim)+"_"\
        +str(args.kernel_size)+"_"+str(args.train_batch_size)+'/'
    args.ck_file_path=output_dir
    print(args.ck_file_path)


def check_point(all_val, model, all_logits,args,eval_score=None):
    output_dir=args.ck_file_path

    seed=args.seed

    if eval_score:
        output_dir+= eval_score +'/'
    os.makedirs(output_dir,  exist_ok=True)

    if args.generate_data:
        filename=output_dir+str(args.datagereate_seed)+"_"+str(seed)+'.pth.tar'
    else:
        filename=output_dir+str(seed)+'.pth.tar'

    if not os.path.exists(filename):
        is_best =True
        save_checkpoint({
        'network':model.state_dict(),
        'logits':all_logits,
        'best_val': all_val,
        'args': args}, is_best,filename)
    else:
        checkpoint = torch.load(filename)
        # import pdb; pdb.set_trace()
        val = checkpoint['best_val'][eval_score]
        best_val=all_val[eval_score]
        is_best =bool(best_val>val)
        if is_best:
            save_checkpoint({
            'network':model.state_dict(),
            'logits':all_logits,
            'best_val': all_val,
            'args': args}, is_best,filename)

if __name__ == "__main__":
    dst='test/'
    copy_file(dst, src=os.getcwd())
