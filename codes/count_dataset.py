#!/usr/bin/env python3
# -*- coding:utf-8 -*-

with open("data/test.txt", encoding='utf8') as f:
    dic = {}
    for line in f:
        line = line.strip()
        e1, e2, relation, sent = line.split("\t")
        if relation not in dic:
            dic[relation] = 1
        else:
            dic[relation] += 1
    print(dic)

# with open("data/raw.txt", encoding='utf8') as f:
#     l = len(f.readlines())
#     print(l)
