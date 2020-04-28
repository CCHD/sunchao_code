#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from utils.string_utils import remove_short_name

# sent = "中国中元国际工程有限公司（以下简称“中国中元”）党委书记、总经理刘小虎一行莅临启迪设计考察交流，启迪设计董事长"
# # sent = remove_short_name(sent)
# entity = "中国中元"
#
# idx = sent.index(entity)
# print(idx)
# idx = sent.find(entity)
# print(idx)

# p = 61.2
# r = 43.2
# f = 2 * p * r / (p + r)
# print(f)


import re
string = "中国中元国际工程有限公司	美国GE通用电气实业有限公司	合作	此外，在战略合作中，中外运空运发展股份有限公司主要负责主要设备的运输，中国中元国际工程有限公司主要负责发电站的设计、施工、安装和项目管理，美国GE通用电气实业有限公司是主要发电机设备的提供商"
en1, en2, relation, sent = string.split("\t")
pt1 = re.compile(en1)
res = pt1.sub("目标企业", sent)
print(res)