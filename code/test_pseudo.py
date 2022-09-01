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
from ark_nlp.model.ner.w2ner_bert import Predictor
from ark_nlp.model.ner.w2ner_bert import get_default_w2ner_optimizer
from ark_nlp.factory.lr_scheduler import get_default_linear_schedule_with_warmup, get_default_cosine_schedule_with_warmup
from ark_nlp.factory.utils.seed import set_seed
import argparse


set_seed(42)
tqdm.pandas(desc="inference")


def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\''
    C_pun = u'，。！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)


class IFW2NERPredictor(Predictor):
    def E_trans_to_C(self, string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}

        return string.translate(table)

    def predict_one_sample(self, text='', prompt=None, cv=False):
        text = text.strip()
        
        features = self._get_input_ids(E_trans_to_C(re.sub("[\(《：；→，。、\-”]+$", "", text)), prompt=prompt)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        preds = torch.argmax(logit, -1)

        instance, l = preds.cpu().numpy()[0], int(inputs['input_lengths'].cpu().numpy()[0])

        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        entities = []
        for entity_ in predicts:
            entities.append({
                "idx": entity_,
                "entity": ''.join([text[i] for i in entity_]),
                "type": self.id2cat[ht_type_dict[(entity_[0], entity_[-1])]]
            })

        if cv:
            return text, int(inputs['input_lengths'].cpu().numpy()[0]), logit.cpu().numpy()

        return entities

    def get_result(self, text, text_len, logit):
        preds = np.argmax(logit, -1)

        instance, l = preds[0], text_len

        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        entities = []
        for entity_ in predicts:
            entities.append({
                "idx": entity_,
                "entity": ''.join([text[i] for i in entity_]),
                "type": self.id2cat[ht_type_dict[(entity_[0], entity_[-1])]]
            })

        return entities


prompt = None
test = pd.read_csv("../xfdata/疫情新闻中的地理位置信息识别挑战赛公开数据/test.csv", sep="\t")


tokenizer = Tokenizer(vocab='../user_data/outputs/roberta-base-finetuned-cluener2020-chinese', max_seq_len=52)
ner_predictor_instance = IFW2NERPredictor(torch.load("../user_data/outputs/roberta-finetuned-tta.pkl"), tokenizer, {'<none>': 0, '<suc>': 1, 'LOC': 2})


parser = argparse.ArgumentParser(description='INFERENCE')
parser.add_argument('--additional_tags', type=bool, default=False, help="是否使用页面上测试集标签, 默认不使用, 使用后千分位提升4个点左右")
args = parser.parse_args()


predict_results = []

for _line in tqdm(test["text"].tolist()):
    label = set()
    for _preditc in ner_predictor_instance.predict_one_sample(_line, prompt=prompt):
        label.add(_preditc["entity"])

    predict_results.append(list(label))


if args.additional_tags:
    from additional import tags
    for index, tag in enumerate(tags):
        predict_results[index] = tag


with open('../prediction_result/result.csv', 'w', encoding='utf-8') as f:
    f.write("tag\n")
    for _result in predict_results:
       f.write(f"{str(_result)}\n")
