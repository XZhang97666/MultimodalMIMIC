import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from module import *
from interp import *
import copy
class BertForRepresentation(nn.Module):

    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name


    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        txt_arr = []

        for input_ids,attention_mask  in zip(input_ids_sequence,attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr



class TextModel(nn.Module):
    def __init__(self,args,device,orig_d_txt=768,Biobert=None):
        """
        Construct a TextModel.
        """
        super(TextModel, self).__init__()

        self.device=device
        self.task=args.task
        self.agg_type=args.agg_type



        self.out_dropout = args.dropout
        self.orig_d_txt=orig_d_txt
        self.d_txt= args.embed_dim
        self.bertrep=BertForRepresentation(args,Biobert)



        self.proj_txt =nn.Linear(self.orig_d_txt, self.d_txt)

        output_dim = args.num_labels




        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if self.task=='ihm':
            self.loss_fct1=CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")


    def forward(self,  input_ids_sequences,
                attn_mask_sequences,labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)

        x_txt=torch.mean(x_txt,dim=1)

        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj)



        if self.task == 'ihm':
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)




class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,orig_d_txt=None,ts_seq_num=None,text_seq_num=None, Biobert=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads


        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text :
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt=orig_d_txt
            self.d_txt= args.embed_dim
            self.text_seq_num=text_seq_num
            self.bertrep=BertForRepresentation(args,Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn=multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels
        if self.modeltype=="TS_Text":
            if self.cross_method=="self_cross":
                self.trans_self_cross_ts_txt=self.get_cross_network(layers=args.cross_layers)
                self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
            else:
                self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
                self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

                if self.cross_method=="MulT":
                    self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
                    self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
                    self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
                elif self.cross_method=="MAGGate":
                    self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.embed_dropout)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                elif  self.cross_method=="Outer":
                    self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                else:
                    self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)


        if self.task=='ihm':
            self.loss_fct1=nn.CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")


    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,self.tt_max, None
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,  self.ts_seq_num,None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt,self.tt_max, None
            else:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim,  q_seq_len,kv_seq_len= self.d_ts,self.tt_max, self.tt_max
            else:

                embed_dim,q_seq_len,kv_seq_len= self.d_ts, self.text_seq_num, self.ts_seq_num
        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                 kv_seq_len=kv_seq_len)

    def get_cross_network(self, layers=-1):
        embed_dim,  q_seq_len= self.d_ts, self.tt_max
        return TransformerCrossEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len)



    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


    def forward(self, x_ts, x_ts_mask, ts_tt_list, input_ids_sequences,
                attn_mask_sequences, note_time_list, note_time_mask_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "TS" in self.modeltype:

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
        if "Text" in self.modeltype:
            x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)
            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_txt=self.time_attn(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)

            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
        if self.cross_method=="self_cross":
            hiddens = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts])
            h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        else:
            if self.cross_method=="MulT":
                # ts --> txt
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                # txt --> ts
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                last_h_ts=proj_x_ts[-1]
                last_h_txt=proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method=="MAGGate":
                    last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
                elif self.cross_method=="Outer":
                    last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)


        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)



        if self.task == 'ihm':
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class TSMixed(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,ts_seq_num=None):

        super(TSMixed, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads

        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.Interp=args.Interp
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.TS_model=args.TS_model
        self.tt_max=args.tt_max


        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_ts=orig_d_ts
        self.d_ts=args.embed_dim
        self.ts_seq_num=ts_seq_num

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2,  self.d_ts, args.embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts=orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level=='batch':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq_feature':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

                # self.moe = nn.Linear(self.d_ts*self.tt_max*2, 1)
        if self.TS_model=='LSTM':
            self.trans_ts_mem=nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts, num_layers=args.layers,dropout=self.dropout,bidirectional=True)

        elif self.TS_model=='CNN':
            self.trans_ts_mem=TimeSeriesCnnModel(input_size=self.d_ts,n_filters=self.d_ts,filter_size=self.kernel_size,\
            dropout=self.dropout,length=self.tt_max,n_neurons=self.d_ts,layers=args.layers)
        elif self.TS_model=='Atten':
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)


        
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer= nn.Linear(self.d_ts, output_dim)

        if self.task=='ihm':
            self.loss_fct1=nn.CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        embed_dim=self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts :
                q_seq_len= self.tt_max
            else:
                q_seq_len= self.ts_seq_num

        return TransformerEncoder(embed_dim=embed_dim,
                                    num_heads=self.num_heads,
                                    layers=layers,
                                    device=self.device,
                                    attn_dropout=self.dropout,
                                    relu_dropout=self.dropout,
                                    res_dropout=self.dropout,
                                    embed_dropout=self.dropout,
                                    attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                    kv_seq_len=None)
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype :

            if self.Interp:
                x_ts_mask_interp=copy.deepcopy(x_ts_mask)
                x_ts_interp=copy.deepcopy(x_ts)
                recon_m=hold_out(x_ts_mask_interp)
                recon_m=torch.Tensor(recon_m).to(self.device)
                proj_x_ts_interp=self.proj_ts_intp(self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m))) #dimension [batch_size,  n_features,seq_len]
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp=self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m, reconstruction=True),reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg=proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts :
                    proj_x_ts_reg=proj_x_ts_interp
                # import pdb; pdb.set_trace()
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                # for name, parameter in self.moe.named_parameters():
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")


            if self.TS_model=='CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)

            elif self.TS_model=='LSTM':
                    _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            if  self.TS_model!='CNN':
                last_h_ts=proj_x_ts[-1]

            else:
                last_h_ts=proj_x_ts

 
            if self.modeltype=="TS" :
                last_hs=last_h_ts
            else:
                raise ValueError("Unknown model type")
                       
    #
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_ts)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)


        if self.Interp:
            reconloss_interp=recon_loss(x_ts_interp,x_ts_mask_interp,recon_m,recon_interp,self.d_ts)

        if self.task == 'ihm':
            if labels!=None:
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)




