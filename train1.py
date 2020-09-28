#!/usr/bin/env python
# coding: utf-8

# # dataset
# 
# + T2Sdataset 能拿到所有資料
#   + get T2Sdataset.train_data, T2Sdataset.dev_data ,T2Sdataset.test_data
# + EncodeModel1 將上述資料轉為ids
#   + get [ encode1 , encode2 ]
# + RandomBatch 模組
#   + 能透過getBatch(size,encode?) 拿到batch data

# In[3]:


import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import argparse
import time
from datetime import datetime, timedelta
import random
import gc
import copy
import os
# get all

class T2Sdataset():
  def __init__(self,train_table_path=None,train_data_path=None,test_table_path=None,test_data_path=None,dev_table_path=None,dev_data_path=None,):
    
    if train_table_path != None:
      self.train_dict = self.get_table(train_table_path)
      self.train_data = self.get_data(self.train_dict,train_data_path)

    if test_table_path != None:
      self.test_dict = self.get_table(test_table_path)
      self.test_data = self.get_data(self.test_dict,test_data_path)
    if dev_table_path != None:
      self.dev_dict = self.get_table(dev_table_path)
      self.dev_data = self.get_data(self.dev_dict,dev_data_path)
  # get the dict of id->header
  def get_table(self,path): 
    id_header = {} 
    with open(path, 'r') as json_file:
      json_list = list(json_file)
    for json_str in json_list:
      result = json.loads(json_str)
      id_header[result['id']] = [result['header'],result['types']]
    return id_header

  def get_data(self,id_header,path):
    data = []
    with open(path, 'r') as json_file:
      json_list = list(json_file)

    for j in json_list:
      j = json.loads(j)
      dic = dict(j)
      dic["header"] = id_header[j['table_id']]
      data.append(dic)
    
    return data


# In[5]:


class RandomBatch():
  def __init__(self,data,model_type):
    
    self.data = sorted(data,key=lambda x:len(x['header'][0])) 
    self.min = len(self.data[0]['header'][0] )# min header length
    self.max = len(self.data[-1]['header'][0] ) # max header length
    self.data_group = [] # 依照header length 分組  
    self.device = torch.device('cuda:0')
    self.tokenizer = BertTokenizer.from_pretrained(model_type)
    self.start()
  def start(self):
    count=[] # 記錄每筆資料長度 [2,2,2,3,3,3,4,4,4,4,5] 
    count_arr = [] # 計算header length 0~22 的有多少 [3,3,4,1]
    for i in self.data:
      count.append(len(i['header'][0]))

    for i in range(self.min,self.max+1):      
      count_arr.append(count.count(i))

    # 機率分布
    self.count_sm = count_arr/np.sum(count_arr)  # [0.01,0.3....0.5]
    current_idx = 0
    for idx in range(len(count_arr)):  
      self.data_group.append(self.data[current_idx:current_idx+count_arr[idx]])
      current_idx += count_arr[idx]  
    return 

  # 一筆資料encode
  def encode(self,data):
    question,headers,sql = data['question'] , data['header'], data['sql'] 
    all_tokens = self.tokenizer.tokenize(question)
    col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}
    for h,t in zip(headers[0],headers[1]):
      tokens = ['[SEP]',col_type_token_dict[t]] + self.tokenizer.tokenize(h)
      all_tokens = all_tokens + tokens

    # get the header token place 
    header_ids = []
    for i in range(len(all_tokens)):
      if all_tokens[i] == col_type_token_dict['text'] or all_tokens[i] == col_type_token_dict['real']:
        header_ids.append(i+1)
    agg = [6]*len(headers[0])
    for s,a in zip(sql['sel'],sql['agg']):
      agg[s] = a
    plus = self.tokenizer.encode_plus(all_tokens,max_length=280,padding='max_length')
    conds_ops = [4]*len(headers[0])
    #conds_vals = ['null']*len(headers[0])
    for i in range(len(sql['conds'])):
      conds = sql['conds'][i] # col type value
      conds_ops[conds[0]] = conds[1]
    for k,v in plus.items():
      plus[k] = torch.tensor(v)
    plus['agg'] = torch.tensor(agg)
    plus['cond_conn_op'] = torch.tensor([sql['cond_conn_op']])
    plus['conds_ops'] = torch.tensor(conds_ops)
    plus['header_ids'] = torch.tensor(header_ids)
    return plus

  # 選出群組
  def selectRandom(self):
    return np.random.choice(np.arange(len(self.count_sm)), size=1, p=self.count_sm).item()

  def list_to_batch(self,data):
    result = {}
    for k in data[0].keys():
      result[k] = torch.stack([i[k] for i in data]).to(self.device)
    result['cond_conn_op'] = result['cond_conn_op'].squeeze()
    return result
  # get batch
  def getBatch(self,batch_size,encode=False):
    datas = self.data_group[self.selectRandom()]
    random_data = random.sample(datas,k=batch_size)
    if encode:
      return self.list_to_batch( [self.encode(i) for i in random_data] )
    else:
      return random_data 


# # Model

# In[7]:


class N2Sm1(nn.Module):
  def __init__(self,model_type):
    super(N2Sm1,self).__init__()
    config = BertConfig.from_pretrained(model_type)
    self.bert_model = BertModel.from_pretrained(model_type,config = config)
    self.cond_conn_op_decoder = nn.Linear(config.hidden_size,3)
    self.agg_deocder = nn.Linear(config.hidden_size,7)
    self.cond_op_decoder = nn.Linear(config.hidden_size,5)

  # 取得header_ids 所標記的 token
  def get_agg_hiddens(self,hiddens,header_ids):
    # header_ids [bsize,headers_idx]
    # hiddens [bsize,seqlength,worddim]
    arr = []
    for b_idx in range(0,hiddens.shape[0]):
      s = torch.stack([hiddens[b_idx][i] for i in header_ids[b_idx]],dim=0)
      arr.append(s)
    return torch.stack(arr,dim=0)

  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,header_ids=None): 
    # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
    hiddens,cls = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    cond_conn_op = self.cond_conn_op_decoder(cls)
    header_hiddens = self.get_agg_hiddens(hiddens,header_ids)
    agg = self.agg_deocder(header_hiddens)
    cond_op = self.cond_op_decoder(header_hiddens)
    # cond_conn_op [bsize,3]   
    # cond_op [bize,header_length,5]
    # agg [bize,header_length,7] 
    return cond_conn_op,cond_op , agg


# # train

# + loss function

# In[8]:


# conn , ops , agg
def getBatchLoss(cond_conn_opX,cond_conn_opY,conds_opsX,conds_opsY,aggX,aggY):

 # print(cond_conn_opX,cond_conn_opY,conds_opsX,conds_opsY,aggX,aggY)

  c = nn.CrossEntropyLoss()
  loss = 0
  loss += c(cond_conn_opX,cond_conn_opY)
  for i in range(2):
    loss += c(conds_opsX[i],conds_opsY[i])
    loss += c(aggX[i],aggY[i])
  #loss = cond_conn_loss + cond_op_loss + agg_loss
  return loss


# In[9]:


def getTime():
  return (datetime.now()+timedelta(hours=8)).strftime("%m/%d %H:%M")


# In[10]:


def train(args):
  print(getTime())
  
  # data
  train_table_file = './data/train/train.tables.json'
  train_data_file = './data/train/train.json'
  val_table_file = './data/val/val.tables.json'
  val_data_file = './data/val/val.json'
  test_table_file = './data/test/test.tables.json'
  test_data_file = './data/test/test.json'
  sample_datas = T2Sdataset(train_table_file,train_data_file,test_table_file,test_data_file,val_table_file,val_data_file)
  rmodel = RandomBatch(sample_datas.train_data,args.model_type)

  # model
  model = N2Sm1(args.model_type).to(args.device)
  
  # parameter & opt
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
  
  minloss = 100000
  counter = 0

  # 1 epoch = random select 20000 time from all data
  # show loss per 4000 times
  for epoch in range(args.epoch):
    print(f'epoch{epoch} {getTime()}')
    epoch_loss = 0
    check_loss = 0
    check_num = 2000
    model.train()
    for r in range(1,10001):
      gc.collect()
      batch = rmodel.getBatch(args.batch_size,encode=True)
      # model input
      input_ids,token_type_ids,attention_mask,header_ids = batch['input_ids'],batch['token_type_ids'],batch['attention_mask'],batch['header_ids']
      # label
      cond_conn_op_label , agg_label , conds_ops_label = batch['cond_conn_op'] , batch['agg'] , batch['conds_ops']
      # pred
      cond_conn_op_pred,conds_ops_pred ,agg_pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,header_ids=header_ids)
      # conn , ops , agg
      batch_loss =  getBatchLoss(cond_conn_op_pred,cond_conn_op_label,conds_ops_pred,conds_ops_label,agg_pred,agg_label)
      epoch_loss += batch_loss
      check_loss += batch_loss
      #print(batch_loss)
      batch_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if r % check_num==0:
        print(f'{check_num} sample AVGLOSS {check_loss/check_num} [{(datetime.now()+timedelta(hours=8)).strftime("%m%d:%H%M")}]')
       # gc.collect()
        check_loss = 0
    print(f'testing...')
    dev_loss = test(model,args,sample_datas)
    print(f'test end loss is {dev_loss}')
    if dev_loss < minloss:
      minloss = dev_loss
      best_model = copy.deepcopy(model.state_dict())
    print(f'Epoches: {epoch} Loss {epoch_loss} AVG: {epoch_loss/20000}')
  return best_model


# # test

# In[11]:


def test(model,args,sample_datas):
  rmodel = RandomBatch(sample_datas.dev_data,args.model_type)

  model.eval()
  total_loss = 0
  round = 1000
  with torch.no_grad():
    for b in range(round):
      batch = rmodel.getBatch(args.batch_size,encode=True)
      # model input
      input_ids,token_type_ids,attention_mask,header_ids = batch['input_ids'],batch['token_type_ids'],batch['attention_mask'],batch['header_ids']
      # label
      cond_conn_op_label , agg_label , conds_ops_label = batch['cond_conn_op'] , batch['agg'] , batch['conds_ops']
      # pred
      cond_conn_op_pred,conds_ops_pred ,agg_pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,header_ids=header_ids)
      # conn , ops , agg
      batch_loss =  getBatchLoss(cond_conn_op_pred,cond_conn_op_label,conds_ops_pred,conds_ops_label,agg_pred,agg_label)
      total_loss += batch_loss
  print(f'Result: LOSS: {total_loss} AVG: {total_loss/round} ')
  return total_loss/round


# # main

# In[12]:



parser = argparse.ArgumentParser([])
parser.add_argument('--batch-size', default=2 , type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--learning-rate', default=1e-5, type=float)    
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--model-type', default='hfl/chinese-bert-wwm' , type=str)  #model_type = 'hfl/chinese-bert-wwm'  'hfl/chinese-roberta-wwm-ext'  'hfl/chinese-roberta-wwm-ext-large'
parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
parser.add_argument('--language', default='china', type=str)
args = parser.parse_args([])
args


# In[1]:



def main():
    mode = 'train'
    if mode == 'train': 
        print('Train')
        best_model = train(args)
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')    
        modelname = 'BertN2S'+'_'+(datetime.now()+timedelta(hours=8)).strftime("%m%d_%H%M")+'.pt' 
        torch.save(best_model, f'saved_models/{modelname}')
        print(f'Train end, model name is {modelname}.pt')
    elif mode == 'test' or mode == 'dev':
        modelname = 'bertDRCD_0807_2222.pt'
        test_model = bertDRCD(args.model_type).to(args.device)
        test_model.load_state_dict(torch.load(f'saved_models/{modelname}'))
        test(test_model,ds,args,mode)


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




