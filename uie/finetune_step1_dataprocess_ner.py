# 数据转换1
import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split


en2ch = {
  'LOC': '提取下述句子中的所有命名实体信息' # '提取文本中包含的所有命名实体'
}


def preprocess(lines, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(save_path, mode + ".json")

    result = []
    tmp = {}
    tmp['id'] = 0
    tmp['text'] = ''
    tmp['relations'] = []
    tmp['entities'] = []

    lines["entities"] = lines.apply(lambda row: [(tag, en2ch["LOC"]) for tag in row["tag"]], axis=1)
    
    # ======= 找出句子中实体的位置 =======
    i = 0
    for text, entity in zip(lines["text"], lines["entities"]):
        if entity:
            ltmp = []
            for ent, type in entity:
                for span in re.finditer(ent, text):
                    start = span.start()
                    end = span.end()
                    ltmp.append((type, start, end, ent))
            ltmp = sorted(ltmp, key=lambda x:(x[1],x[2]))
            for j in range(len(ltmp)):
                tmp['entities'].append({"id":j, "start_offset":ltmp[j][1], "end_offset":ltmp[j][2], "label":ltmp[j][0]})
        else:
            tmp['entities'] = []

        tmp['id'] = i
        tmp['text'] = text
        result.append(tmp)
        tmp = {}
        tmp['id'] = 0
        tmp['text'] = ''
        tmp['relations'] = []
        tmp['entities'] = []
        i += 1

    with open(data_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in result]))


def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\''
    C_pun = u'，。！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)


data = pd.read_csv("/data/lpzhang/ner/data/train.csv", sep="\t")
data["text"] = data["text"].apply(lambda line: E_trans_to_C(re.sub("[\(《：→；，。、\-”]+$", "", line.strip())))
data["tag"] = data["tag"].apply(lambda x: [E_trans_to_C(i) for i in eval(str(x))])

tta = pd.read_csv("/data/lpzhang/ner/data/tta_all.csv", sep="\t")
tta["text"] = tta["text"].apply(lambda line: E_trans_to_C(re.sub("[\(《：→；，。、\-”]+$", "", line.strip())))
tta["tag"] = tta["tag"].apply(lambda x: [E_trans_to_C(i) for i in eval(str(x))])

train, dev = train_test_split(data, test_size=0.3)

preprocess(pd.concat([train, tta]).reset_index(drop=True), './mid_data', "train_tta")
preprocess(dev.reset_index(drop=True), './mid_data', "dev_tta")
