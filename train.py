
from checkpoint import *
from until import *
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score, precision_recall_curve,  auc, f1_score

def eval_test(args,model,test_data_loader,device):
    model.eval()
    rootdir=args.ck_file_path

    result_file=rootdir
    os.makedirs(rootdir,exist_ok = True)

    try:
        result_dict = pickle.load(open(rootdir+"result.pkl", "rb"))
    except:
        result_dict={}


    if args.generate_data:
        seed=str(args.datagereate_seed)+'_'+str(args.seed)
    else:
        seed=args.seed
    result_dict[seed]={}
    for subdir, dirs, files in os.walk(rootdir):
        if len(files)==0 or 'pkl' in files[0]:
            continue
        substr=subdir.split('/')[-1]
        if substr=='model':
            continue

        file= str(seed)+'.pth.tar'
        file_path=os.path.join(subdir, file)
        print(file_path)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['network'])
        test_val=evaluate_irg(args=args, device=device,data_loader=test_data_loader,model=model,mode='test' )
        for eval_type, val in test_val.items():
            result_dict[seed][eval_type]={}
            result_dict[seed][eval_type]['val']=checkpoint['best_val'][eval_type]
            result_dict[seed][eval_type]['test']=test_val[eval_type]


    with open(rootdir+"/result.pkl","wb") as f:
        pickle.dump(result_dict, f)


def trainer_irg (model,args,accelerator,train_dataloader,dev_dataloader,test_data_loader,device,optimizer,pretrain_epoch=None,writer=None,scheduler=None):
    count=0
    global_step=0
    best_evals={}
    for epoch in tqdm(range(args.num_train_epochs)):
        model.train()
        if "Text" in args.modeltype:
            # import pdb;
            # pdb.set_trace()
            if args.num_update_bert_epochs<args.num_train_epochs and (epoch)%args.num_update_bert_epochs==0 and count<args.bertcount:
                count+=1
                print("bert update at epoch "+ str(epoch) )
                for param in model.bertrep.parameters():
                        param.requires_grad = True
            else:
                for param in model.bertrep.parameters():
                    param.requires_grad = False
        # import pdb;
        # pdb.set_trace()

            for param in model.bertrep.parameters():
                print(epoch,param.requires_grad)
                break



        for step, batch in tqdm(enumerate(train_dataloader)):
            global_step+=1

            ts_input_sequences,ts_mask_sequences, ts_tt, reg_ts , input_ids_sequences,attn_mask_sequences, note_time ,note_time_mask, label = batch

            if "Text" not in args.modeltype:
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        labels=label,reg_ts=reg_ts)

            else:
                loss=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,labels=label,reg_ts=reg_ts)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)



            if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                if scheduler!=None:
                    scheduler.step()
                model.zero_grad()

            if writer!=None:
                writer.add_scalar('training/train_loss',loss,global_step)
        eval_vals=evaluate_irg(args,device,dev_dataloader,model)
        for k,v in eval_vals.items():
            if k== 'auc_scores':
              continue
            if writer!=None:
                writer.add_scalar('dev/'+k ,v,epoch+1)
            best_eval=best_evals.get(k, 0)
            if v>best_eval:
                best_eval=v
                best_evals[k]=best_eval
            print("Current "+ k,v)
            print("Best "+ k,best_eval)

        if writer!=None:
            writer.close()



def evaluate_irg(args,device,data_loader,model,mode=None):
    model.eval()
    # c=0
    eval_logits= []
    eval_example = []
    for idx,batch in enumerate(tqdm(data_loader)):

        ts_input_sequences,ts_mask_sequences, ts_tt, reg_ts,input_ids_sequences,attn_mask_sequences, note_time ,note_time_mask, label  = batch


        with torch.no_grad():

            if "Text" not in args.modeltype:
                logits=model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        reg_ts=reg_ts)

            else:
                
                logits = model(x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,reg_ts=reg_ts)
            if torch.isnan(logits).any():
                import pdb;
                pdb.set_trace()
            logits=logits.cpu().numpy()
            label_ids=label.cpu().numpy()
            eval_logits += logits.tolist()
            eval_example +=label_ids.tolist()


    eval_vals={}

    # import pdb;
    # pdb.set_trace()

    all_logits = np.array(eval_logits)
    all_label = np.array(eval_example)
    all_pred= np.where(all_logits > 0.5, 1, 0)
    if args.task=="pheno":
        eval_vals=metrics_multilabel(all_label, all_logits, verbose=0)
        eval_vals['macro_f1']=f1_score(all_label, all_pred, average='macro')
        # import pdb; pdb.set_trace()
        if mode==None:
            check_point(eval_vals, model, eval_logits, args,"macro_f1")

    elif args.task=='ihm':
            eval_val =roc_auc_score(np.array(eval_example), np.array(eval_logits))
            eval_vals['auc']=eval_val
            (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_example), np.array(eval_logits))
            eval_val = auc(recalls, precisions)
            eval_vals['auprc']=eval_val
            eval_val=f1_score(np.array(eval_example), all_pred)
            eval_vals['f1']=eval_val
            if mode==None:
                check_point(eval_vals, model, eval_logits, args,"f1")

    return eval_vals
