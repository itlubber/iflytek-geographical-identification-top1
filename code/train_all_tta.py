import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(".")

import re
import os
import jieba
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from ark_nlp.model.ner.w2ner_bert import W2NERBertConfig
from ark_nlp.model.ner.w2ner_bert import Tokenizer
from ark_nlp.model.ner.w2ner_bert import W2NERBert
from ark_nlp.model.ner.w2ner_bert import Dataset
from ark_nlp.model.ner.w2ner_bert import Task
from ark_nlp.model.ner.w2ner_bert import get_default_w2ner_optimizer
from ark_nlp.factory.lr_scheduler import get_default_linear_schedule_with_warmup, get_default_cosine_schedule_with_warmup
from ark_nlp.factory.utils.seed import set_seed


set_seed(42)
tqdm.pandas(desc="finetune")


def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\''
    C_pun = u'，。！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)


train = pd.read_csv("../xfdata/疫情新闻中的地理位置信息识别挑战赛公开数据/train.csv", sep="\t")

# 伪标签数据
test = pd.read_csv("../xfdata/疫情新闻中的地理位置信息识别挑战赛公开数据/test.csv", sep="\t")
test["tag"] = pd.read_csv("../prediction_result/result.csv", sep="\t")["tag"]
test["text"] = test["text"].apply(lambda line: E_trans_to_C(re.sub("[\(《：→；，。、\-”]+$", "", line.strip())))
test["tag"] = test["tag"].apply(lambda x: [E_trans_to_C(i) for i in eval(str(x))])
test = test[test["tag"].apply(len) > 0]


train = pd.concat([train[["text", "tag"]], test[["text", "tag"]]]).reset_index(drop=True)


train["text"] = train["text"].apply(lambda line: E_trans_to_C(re.sub("[\(《：→；，。、\-”]+$", "", line.strip())))
train["tag"] = train["tag"].apply(lambda x: [E_trans_to_C(i) for i in eval(str(x))])


train["entities"] = train.progress_apply(lambda row: [["LOC", *i.span()] for tag in row["tag"] for i in re.finditer(tag, row["text"])], axis=1)


datalist = []

for _, row in train.iterrows():
    entity_labels = []
    for _type, _start_idx, _end_idx in row["entities"]:
        entity_labels.append({
            'start_idx': _start_idx,
            'end_idx': _end_idx,
            'type': _type,
            'entity': row["text"][_start_idx: _end_idx]
    })

    datalist.append({
        'text': row["text"],
        'entities': entity_labels
    })

train_data_df = pd.DataFrame(datalist)


def get_label(x):
    
    entities = []
    for entity in x:
        entity_ = {}
        idx = list(range(entity['start_idx'], entity['end_idx']))
        entity_['idx'] = idx
        entity_['type'] = entity['type']
        entity_['entity'] = entity['entity']
        entities.append(entity_)
    
    return entities


train_data_df['label'] = train_data_df['entities'].apply(lambda x: get_label(x))
train_data_df = train_data_df.loc[:, ['text', 'label']]
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))


ner_train_dataset = Dataset(train_data_df)


promot = None
tokenizer = Tokenizer(vocab='../user_data/roberta-base-finetuned-cluener2020-chinese', max_seq_len=52)
ner_train_dataset.convert_to_ids(tokenizer, promot=promot)


config = W2NERBertConfig.from_pretrained('../user_data/roberta-base-finetuned-cluener2020-chinese', num_labels=len(ner_train_dataset.cat2id))
torch.cuda.empty_cache()
dl_module = W2NERBert.from_pretrained('../user_data/roberta-base-finetuned-cluener2020-chinese', config=config)

# 设置运行次数
num_epoches, batch_size = 40, 256
optimizer = get_default_w2ner_optimizer(dl_module, lr=1e-2, bert_lr=5e-5, weight_decay=0.01)


# 注意lr衰减轮次的设定
show_step = len(ner_train_dataset) // batch_size + 2
t_total = len(ner_train_dataset) // batch_size * num_epoches
scheduler = get_default_cosine_schedule_with_warmup(optimizer, t_total, warmup_ratio=0.2)


model = Task(dl_module, optimizer, 'ce', cude_device=2, scheduler=scheduler, grad_clip=5.0, ema_decay=0.995, fgm_attack=True, save_path="../user_data/outputs/roberta-finetuned-tta", )


model.fit(ner_train_dataset, epochs=num_epoches, batch_size=batch_size, show_step=show_step)
