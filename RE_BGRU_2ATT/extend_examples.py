#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os


def select_example(confidence_threshold=0.2):
    file_list = ["to_extend_0-100.txt", "to_extend_100-200.txt", "to_extend_200-300.txt", "to_extend_300-400.txt",
                 "to_extend_400-500.txt", "to_extend_500-600.txt", "to_extend_600-700.txt"]
    example_to_extend = []
    for i, filename in enumerate(file_list):
        if i == 0 or i == 1:
            pass
        else:
            continue
        print("扩充数据文件：{}-{}".format(i, filename))
        with open(os.path.join("origin_data_record/example_with_confidence", filename), encoding='utf8') as f:
            for line in f:
                en1, en2, relation, confidence, sentence = line.strip().split('\t')
                confidence = float(confidence)
                if confidence < confidence_threshold:
                    continue
                string = "{}\t{}\t{}\t{}\n".format(en1, en2, relation, sentence)
                example_to_extend.append(string)
    print("扩充数据数量：{}".format(len(example_to_extend)))
    output_file = "origin_data_record/example_to_extend.txt"
    with open(output_file, 'w', encoding='utf8') as fo:
        for string in example_to_extend:
            fo.write(string)
    print("样本保存到：{}".format(output_file))


select_example()