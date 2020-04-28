#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import random
import re
from utils.string_utils import remove_short_name, remove_num, remove_space


def make_bags():
    labels = []
    labels_dic = {}
    relation_dic = {}
    for root, dirnames, filenames in os.walk("data/train"):
        for filename in filenames:
            with open(os.path.join("data/train", filename), encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    dic = json.loads(line)
                    raw_types = dic['raw_type']
                    sentence = dic['sentence']

                    entity1 = dic['entity1']
                    for i, raw_type in enumerate(raw_types):
                        if raw_type not in labels:
                            labels.append(raw_type)
                            labels_dic[raw_type] = 1
                        else:
                            labels_dic[raw_type] += 1
                        if raw_type == "":
                            continue
                        elif raw_type == "交流" or raw_type == "合作意向":
                            relation = "合作意向"
                        elif raw_type == "组建" or raw_type == "控股" or raw_type == "持有股权" or raw_type == "子公司" or raw_type == "成立" or raw_type == "隶属" or raw_type == "从属" or raw_type == "投资" or raw_type == "持有股份":
                            relation = "持有股权"
                        elif raw_type == "签订协议" or raw_type == "签署协议" or raw_type == "合作伙伴" or raw_type == "战略合作" or raw_type == "合资" or raw_type == "合作建设" or raw_type == "共同发起" or raw_type == "共同承办" or raw_type == "联合开发" or raw_type == "联合打造" or raw_type == "合作" or raw_type == "共同成立" or raw_type == "共同建设":
                            relation = "合作"
                        elif raw_type == "收购股权" or raw_type == "收购":
                            relation = "收购股权"
                        elif raw_type == "其他" or raw_type == "更名" or raw_type == "出让股权":
                            relation = "其他"
                        entity2 = dic['entity2'][i]
                        if relation not in relation_dic:
                            relation_dic[relation] = []
                        string = entity1 + "\t" + entity2 + "\t" + relation + "\t" + sentence + "\n"
                        relation_dic[relation].append(string)

    with open("data/relation_count", 'w', encoding='utf8') as fo:
        for k, v in labels_dic.items():
            fo.write(k + " " + str(v) + "\n")
    print(labels_dic)
    print(relation_dic)
    for k, v in relation_dic.items():
        with open(os.path.join("data/bags", k), 'w', encoding='utf8') as fo:
            for line in v:
                # fo.write(line)
                temp = remove_short_name(line)
                fo.write(temp)

    return labels


def replace_entity_in_bags():
    for root, dirnames, filenames in os.walk("data/bags"):
        for filename in filenames:
            new_lines = []
            with open(os.path.join("data/bags", filename), encoding='utf8') as f:
                for line in f:
                    en1, en2, relation, sent = line.strip().split("\t")
                    sent = sent.strip()
                    sent = re.sub(en1, "目标实体", sent)
                    sent = re.sub(en2, "相关实体", sent)
                    new_line = "{}\t{}\t{}\t{}\n".format("目标实体", "相关实体", relation, sent)
                    print(new_line)
                    new_lines.append(new_line)
            with open(os.path.join("data/new_bags", filename), 'w', encoding='utf8') as fo:
                for line in new_lines:
                    fo.write(line)
    return


def make_re_dataset():
    train_set = []
    test_set = []
    for root, dirnames, filenames in os.walk("data/bags"):
        for filename in filenames:
            if filename == "其他":
                continue
            with open(os.path.join("data/bags/", filename), encoding='utf8') as f:
                all_sent = []
                for line in f:
                    all_sent.append(line)
                random.shuffle(all_sent)
                l = len(all_sent)
                threshold = int(l * 0.8)
                train_set += all_sent[:threshold]
                test_set += all_sent[threshold:]
    with open("data/train.txt", 'w', encoding='utf8') as fo:
        for sent in train_set:
            fo.write(sent)
    with open("data/test.txt", 'w', encoding='utf8') as fo:
        for sent in test_set:
            fo.write(sent)


def make_ner_dataset():
    all_sents = []
    all_tags = []
    for root, dirnames, filenames in os.walk("data/train"):
        for filename in filenames:
            with open(os.path.join("data/train", filename), encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    dic = json.loads(line)
                    entity2 = dic['entity2']
                    if not entity2:
                        continue
                    sentence = dic['sentence']
                    sentence = remove_space(sentence)
                    sentence = remove_short_name(sentence)
                    if "　" in sentence:
                        print(sentence)
                    words = list(sentence)
                    entity1 = dic['entity1']
                    tags = ['O'] * len(sentence)
                    for entity in (entity2 + [entity1]):
                        if not entity:
                            continue
                        idx = sentence.find(entity)
                        if idx != -1:
                            tags[idx] = 'B-ORG'
                            for i in range(idx+1, idx+len(entity)):
                                tags[i] = 'I-ORG'
                    all_sents.append(words)
                    all_tags.append(tags)
    with open("data/ner/train_data", 'w', encoding='utf8') as fo_train:
        with open("data/ner/test_data", 'w', encoding='utf8') as fo_test:
            for i in range(len(all_sents)):
                words = all_sents[i]
                tags = all_tags[i]
                r = random.randint(0, 99)
                fo = fo_train
                if r < 15:
                    fo = fo_test
                for j in range(len(words)):
                    fo.write("{} {}\n".format(words[j], tags[j]))
                fo.write('\n')


if __name__ == '__main__':
    # make_bags()
    # replace_entity_in_bags()
    # make_re_dataset()
    make_ner_dataset()
