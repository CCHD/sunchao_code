#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re

space_pt = re.compile("[   \t\n　]*")
short_name_pt = re.compile("[(（]以下简称.*?[)）]")

pt1 = re.compile("\(.*?\)")
pt2 = re.compile("（.*?）")
num_pt = re.compile("[0-9]*")


def remove_space(string):
    string = space_pt.sub("", string)
    return string


def remove_short_name(string):
    string = pt1.sub("", string)
    string = pt2.sub("", string)
    return string


def remove_num(string):
    string = num_pt.sub("", string)
    return string


if __name__ == "__main__":
    # sent = "12月中旬，由中国中元国际工程有限公司能源与环境工程设计研究院才所长一行3人到廊坊市 晋盛节能 技术服务有限公司进行了交流。"
    # sent = remove_space(sent)
    # print(sent)

    sent2 = "4月15日，中国中元国际工程有限公司（以下简称“中国中元”）党委书记、总经理刘小虎一()（）行莅临启迪设计考察交流，"
    sent2_res = remove_short_name(sent2)
    print(sent2_res)