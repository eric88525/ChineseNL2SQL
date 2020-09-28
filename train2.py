
# 訓練方式為 col > [各種舉例數字]

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import cn2an
import random
import gc
import copy
import os
import argparse

import time
from datetime import datetime, timedelta

"""# 數字抽取"""

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string

def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string

def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:   
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None

def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None
    
def load_json(json_file):
    result = []
    if json_file:
        with open(json_file) as file:
            for line in file:
                result.append(json.loads(line))
    return result

def extract_values_from_text(text):
        values = []
        values += extract_year_from_text(text)
        values += extract_num_from_text(text)
        return list(set(values))

def extract_num_from_text(text):
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    values = []
    num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
    values += num_values
        
    cn_num_unit = CN_NUM +CN_UNIT
    cn_num_texts = re.findall(r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
    cn_num_values = [str_to_num(text) for text in cn_num_texts]
    values += [value for value in cn_num_values if value is not None]
    
    cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(CN_UNIT), text)
    for word in cn_num_mix:
        num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
        for n in num:
            word = word.replace(n, an_to_cn(n))
        str_num = str_to_num(word)
        if str_num is not None:
            values.append(str_num)
    return values


def extract_year_from_text(text):
  values = []
  CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
  CN_UNIT = '十拾百佰千仟万萬亿億兆点'
  num_year_texts = re.findall(r'[0-9][0-9]年', text)
  values += ['20{}'.format(text[:-1]) for text in num_year_texts]
  cn_year_texts = re.findall(r'[{}][{}]年'.format(CN_NUM,CN_NUM), text)
  cn_year_values = [str_to_year(text) for text in cn_year_texts]
  values += [value for value in cn_year_values if value is not None]
  return values

"""# dataset

+ T2Sdataset 能拿到所有資料
  + get T2Sdataset.train_data, T2Sdataset.dev_data ,T2Sdataset.test_data
+ EncodeModel1 將上述資料轉為ids
  + get [ encode1 , encode2 ]
+ RandomBatch 模組
  + 能透過getBatch(size,encode?) 拿到batch data
"""

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
      id_header[result['id']] = [result['header'],result['types'],result['rows']]
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



class Model2Encoder(Dataset): # real類型瓊舉
    def __init__(self,data,data_table,model_type,device):
      self.tokenizer = BertTokenizer.from_pretrained(model_type)
      self.device = device
      self.data = self.make_pairs(data,data_table)
      
    # 將原始的一筆資料拆分成多筆
    def make_pairs(self,data,data_table):
      all_pairs = []
      operator_list = ['>','<','=','!=']     
      for d in data:
        table = data_table[d['table_id']] # a list [[headername],[headertype],[[row1],[row2]]]
        conds = d['sql']['conds'] # a list like [[0, 2, '大黄蜂'], [0, 2, '密室逃生']]
        for col,op,value in conds:
          temp = []
          if len(value)>300:
            continue
          #question_ids = self.tokenizer.tokenize(d['question']) + ['[SEP]']         
          label = 0
          if table[1][col] == 'real':
            cname = table[0][col]
            values_in_q = extract_values_from_text(d['question']) # 問句中的數字
            for v in values_in_q:
              if str(v) == str(value):
                temp.append([ d['question'] , cname + operator_list[op] + str(v) ,1] )
              else:
                temp.append([ d['question'] , cname + operator_list[op] + str(v) ,0] )
         
            all_pairs+=temp
          elif table[1][col] == 'text': # text
            cname = table[0][col]
            candidate_list = (row[col] for row in table[2]) # 一堆=右邊的值 像是 大黄蜂 密室逃生
            # 1 放進去
            all_pairs.append([ d['question'] , cname+'='+value ,1]) 
            for ci,cl in enumerate(candidate_list):
              if len(cl)>300 or cl == value:
                continue
              if ci>3:
                break
              all_pairs.append([ d['question'] , cname+'='+cl ,label]) 
              #all_pairs.append([text_id,label])
      return all_pairs
    def __getitem__(self,idx):
      ids =  self.tokenizer.encode_plus(self.data[idx][0],self.data[idx][1],max_length=270,padding='max_length')
      ids['label'] = torch.Tensor([self.data[idx][2]]).to(self.device)
      for k in ids.keys():
        ids[k] = torch.tensor(ids[k]).to(self.device)
      return ids
    def __len__(self):
      return len(self.data)

"""# Model"""

class N2Sm2(nn.Module):
  def __init__(self,model_type):
    super(N2Sm2,self).__init__()
    config = BertConfig.from_pretrained(model_type)
    self.bert_model = BertModel.from_pretrained(model_type,config = config)
    self.decoder = nn.Sequential(
        nn.Linear(config.hidden_size,1),
        nn.Sigmoid()
    )
    
  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,header_ids=None): 
    # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
    hiddens,cls = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    return self.decoder(cls)


"""# train

+ loss function
"""

def getTime():
  return (datetime.now()+timedelta(hours=8)).strftime("%m/%d %H:%M")

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
  # !!!
  sample_data = Model2Encoder(sample_datas.train_data ,sample_datas.train_dict,args.model_type,torch.device('cuda:0'))
  trainLoader = DataLoader(sample_data, batch_size=args.batch_size  ,shuffle=True)
  # model
  model = N2Sm2(args.model_type).to(args.device)  
  # parameter & opt
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay) 
  minloss = 100000
  counter = 0
  cri = nn.BCELoss()
  # 1 epoch = random select 20000 time from all data
  # show loss per 4000 times
  for epoch in range(args.epoch):
    print(f'epoch{epoch} {getTime()}')
    epoch_loss = 0
    check_loss = 0
    check_num = args.check_num
    model.train()
    for r,batch in enumerate(trainLoader):
      gc.collect()
      input_ids,token_type_ids,attention_mask,label = batch['input_ids'],batch['token_type_ids'],batch['attention_mask'],batch['label']
      pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)     
      batch_loss = cri(pred,label)
      epoch_loss += batch_loss
      check_loss += batch_loss
      batch_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if (r+1) % check_num==0:
        print(f'{check_num} sample AVGLOSS {check_loss/check_num} [{(datetime.now()+timedelta(hours=8)).strftime("%m%d:%H%M")}]')      
        check_loss = 0
    print(f'testing...')
    try:
      dev_loss = -1
      #dev_loss = test(model,args,sample_datas)
      #print(f'test end loss is {dev_loss}')
      if dev_loss <= minloss:
        minloss = dev_loss
        best_model = copy.deepcopy(model.state_dict())
    except:
      best_model = copy.deepcopy(model.state_dict())
    print(f'Epoches:{epoch} Loss {epoch_loss}')
  return best_model



"""# test"""

def test(model,args,sample_datas):
  sample_data = Model2Encoder(sample_datas.dev_data ,sample_datas.dev_dict,args.model_type,torch.device('cuda:0'))
  devLoader = DataLoader(sample_data, batch_size=args.batch_size ,shuffle=True)
  cri = nn.BCELoss()
  model.eval()
  total_loss = 0
  round = 0
  with torch.no_grad():
    for batch in devLoader:
      round = round+1
      input_ids,token_type_ids,attention_mask,label = batch['input_ids'],batch['token_type_ids'],batch['attention_mask'],batch['label']
      pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)         
      batch_loss = cri(pred,label)
      total_loss+=batch_loss
  print(f'Result: LOSS: {total_loss} AVG: {total_loss/round}')
  return total_loss/round

"""# Check acc

# main
"""

parser = argparse.ArgumentParser([])
parser.add_argument('--batch-size', default=16, type=int)
#check_num
parser.add_argument('--check-num', default=2000,type=int)
parser.add_argument('--epoch', default=5, type=int)
parser.add_argument('--learning-rate', default=1e-5, type=float)    
parser.add_argument('--weight-decay', default=0.001, type=float)
parser.add_argument('--model-type', default='hfl/chinese-bert-wwm' , type=str)  #model_type = 'hfl/chinese-bert-wwm'  'hfl/chinese-roberta-wwm-ext'  'hfl/chinese-roberta-wwm-ext-large'
parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
args = parser.parse_args([])

mode = 'train'
if mode == 'train': 
  if not os.path.exists('saved_models'):
    os.makedirs('saved_models')    
  print('Train')
  best_model = train(args) 
  modelname = 'BertN2S_M2'+'_'+(datetime.now()+timedelta(hours=8)).strftime("%m%d_%H%M")+'.pt' 
  torch.save(best_model, f'saved_models/{modelname}')
  print(f'Train end, model name is {modelname}')
elif mode == 'acc':
  print(1)