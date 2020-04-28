#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import json
import os

from pymongo import MongoClient

from utils.string_utils import remove_space


def download_raw():
    client = MongoClient(host='192.168.1.27', port=27017)
    dblist = client.database_names()
    db = client.news
    coll = db.ai_news
    for i, docu in enumerate(coll.find({"content": {"$regex": ".*合作.*"}, "source": "萝卜投研"})):
        print(i)
        if i <= 8276:
            continue
        dic = {
            # "_id": docu['_id'],
            "url": docu['url'],
            "content": docu['content'],
            "entity1": docu['full_key'],
            "entity2": "",
            "relation_type": ""
        }
        string = json.dumps(dic, ensure_ascii=False, indent=4)
        with open("data/raw/" + str(i) + ".json", 'w', encoding='utf8') as fo:
            fo.write(string)


def multi_file_to_one():
    string_list = []
    for i in range(0, 54019):
        with open(os.path.join("data/raw", str(i) + '.json'), encoding='utf8') as f:
            string = f.read().strip()
            dic = json.loads(string)
            dic['idx'] = i
            string = json.dumps(dic, ensure_ascii=False)
            string_list.append(string)
        print(i)
    with open("data/raw.txt", 'w', encoding="utf8") as fo:
        for s in string_list:
            fo.write(s)
            fo.write("\n")


def make_train_set():
    with open("data/raw.txt", encoding='utf8') as f:
        for i, line in enumerate(f.readlines()):
            if 0 <= i < 516:
                continue

            dic = json.loads(line.strip())
            content = dic['content']
            entity1 = dic['entity1']

            sentences = remove_space(content).split("。")
            print("idx: ", i, dic['idx'])

            records = []
            for sent in sentences:
                if entity1 in sent:
                    print("Sentence: ", sent)
                    print("Entity1: ", entity1)

                    entity2 = input("Entity: ").split(" ")
                    raw_type = input("Raw relation type: ").split(" ")
                    temp = {
                        "sentence": sent,
                        "entity1": entity1,
                        "relation_type": "",
                        "entity2": entity2,
                        "raw_type": raw_type}
                    records.append(temp)
            with open(os.path.join("data/train", str(i) + ".json"), 'w', encoding='utf8') as fo:
                for record in records:
                    temp_string = json.dumps(record, ensure_ascii=False)
                    fo.write(temp_string)
                    fo.write("\n")
            # break



# multi_file_to_one()
make_train_set()