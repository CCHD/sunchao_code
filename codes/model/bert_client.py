#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from bert_serving.client import BertClient


bc = BertClient()

temp = bc.encode(['我 喜欢 你们', '我 喜 欢 你 们'])


print(temp)
