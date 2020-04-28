from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import os
import network

FLAGS = tf.app.flags.FLAGS


def get_cos_distance(X1, X2):
    # calculate cos distance between two sets
    # more similar more big
    with tf.Session() as sess:
        (k,n) = X1.shape
        (m,n) = X2.shape
        # 求模
        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
        # 内积
        X1_X2 = tf.matmul(X1, tf.transpose(X2))
        X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[k,1]),tf.reshape(X2_norm,[1,m]))
        # 计算余弦距离
        cos = X1_X2/X1_X2_norm
        res = sess.run(cos)
    return res


def cosine(q, a):
    with tf.Session() as sess:
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        res = sess.run(score)
    return res


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# If you retrain the model, please remember to change the path to your own model below:
pathname = "./model/ATT_GRU_model-2700"

wordembedding = np.load('./data/vec.npy')
test_settings = network.Settings()
test_settings.vocab_size = 16693
test_settings.num_classes = 5
test_settings.big_num = 1

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        def predict_step(word_batch, pos1_batch, pos2_batch, y_batch):

            feed_dict = {}
            total_shape = []
            total_num = 0
            total_word = []
            total_pos1 = []
            total_pos2 = []

            for i in range(len(word_batch)):
                total_shape.append(total_num)
                total_num += len(word_batch[i])
                for word in word_batch[i]:
                    total_word.append(word)
                for pos1 in pos1_batch[i]:
                    total_pos1.append(pos1)
                for pos2 in pos2_batch[i]:
                    total_pos2.append(pos2)

            total_shape.append(total_num)
            total_shape = np.array(total_shape)
            total_word = np.array(total_word)
            total_pos1 = np.array(total_pos1)
            total_pos2 = np.array(total_pos2)

            feed_dict[mtest.total_shape] = total_shape
            feed_dict[mtest.input_word] = total_word
            feed_dict[mtest.input_pos1] = total_pos1
            feed_dict[mtest.input_pos2] = total_pos2
            feed_dict[mtest.input_y] = y_batch

            loss, accuracy, prob, feature_map = sess.run(
                [mtest.loss, mtest.accuracy, mtest.prob, mtest.attention_r], feed_dict)
            return prob, accuracy, feature_map

        with tf.variable_scope("model"):
            mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

        names_to_vars = {v.op.name: v for v in tf.global_variables()}
        saver = tf.train.Saver(names_to_vars)
        saver.restore(sess, pathname)

        print('reading word embedding data...')
        vec = []
        word2id = {}
        f = open('./origin_data/vec.txt', encoding='utf-8')
        content = f.readline()
        content = content.strip().split()
        dim = int(content[1])
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split()
            word2id[content[0]] = len(word2id)
            content = content[1:]
            content = [(float)(i) for i in content]
            vec.append(content)
        f.close()
        word2id['UNK'] = len(word2id)
        word2id['BLANK'] = len(word2id)

        print('reading relation to id')
        relation2id = {}
        id2relation = {}
        f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split()
            relation2id[content[0]] = int(content[1])
            id2relation[int(content[1])] = content[0]
        f.close()


# todo
def predict_example(en1, en2, sentence):
    # print("实体1: " + en1)
    # print("实体2: " + en2)
    # print(sentence)

    en1pos = sentence.find(en1)
    if en1pos == -1:
        en1pos = 0
    en2pos = sentence.find(en2)
    if en2pos == -1:
        en2pos = 0
    output = []
    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    #Encoding test x
    for i in range(fixlen):
        word = word2id['BLANK']
        rel_e1 = pos_embed(i - en1pos)
        rel_e2 = pos_embed(i - en2pos)
        output.append([word, rel_e1, rel_e2])

    for i in range(min(fixlen, len(sentence))):

        word = 0
        if sentence[i] not in word2id:
            #print(sentence[i])
            #print('==')
            word = word2id['UNK']
            #print(word)
        else:
            #print(sentence[i])
            #print('||')
            word = word2id[sentence[i]]
            #print(word)

        output[i][0] = word
    test_x = []
    test_x.append([output])

    #Encoding test y
    label = [0 for i in range(len(relation2id))]
    label[0] = 1
    test_y = []
    test_y.append(label)

    test_x = np.array(test_x)
    test_y = np.array(test_y)


    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(test_x)):
        word = []
        pos1 = []
        pos2 = []
        for j in test_x[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)



    prob, accuracy, feature_map = predict_step(test_word, test_pos1, test_pos2, test_y)
    prob = np.reshape(np.array(prob), (1, test_settings.num_classes))[0]

    return prob, feature_map


def bags_to_feature():
    bags = {
        "收购股权": [],
        "合作": [],
        "合作意向": [],
        "持有股权": [],
        "其他": []
    }
    for relation_name in bags.keys():
        with open(os.path.join("origin_data_record/new_bags", relation_name), encoding='utf8') as file_for_bags:
            for line in file_for_bags:
                en1, en2, relation, sentence = line.strip().split('\t')
                prob, feature = predict_example(en1, en2, sentence)
                bags[relation_name].append(feature)
    # print(bags)
    return bags


def calculate_confidence(feature_list, example_feature):
    scores = []
    for feature in feature_list:
        scores.append(get_cos_distance(feature, example_feature))
    confidence = sum(scores) / len(scores)
    return confidence


def store_example_confidence():
    start = 600
    end = 700

    bags = bags_to_feature()

    unlabeled = []
    with open("origin_data_record/unlabeled.txt", encoding='utf8') as f:
        for line in f:
            unlabeled.append(line)
    to_extend = []
    for i, line in enumerate(unlabeled):
        if start <= i < end:
           pass
        else:
            continue
        start_time = time.time()
        en1, en2, relation, sentence = line.strip().split('\t')

        prob, feature = predict_example(en1, en2, sentence)
        top3_id = prob.argsort()[-3:][::-1]
        relation = id2relation[top3_id[0]]
        if relation == "其他":
            continue
        confidence = calculate_confidence(bags[relation], feature)
        # print("实体1：", en1)
        # print("实体2：", en2)
        # print("关系：", relation)
        # print("原文：", sentence)
        # print("置信度：", confidence)
        print("编号：\t", i)
        print("置信度：\t", confidence[0][0])
        print("运行时间：\t", time.time() - start_time)
        string = "{}\t{}\t{}\t{}\t{}\n".format(en1, en2, relation, confidence[0][0], sentence)
        to_extend.append(string)
    with open("origin_data_record/example_with_confidence/to_extend_{}-{}.txt".format(start, end), 'w', encoding='utf8') as fo:
        for string in to_extend:
            fo.write(string)


def main():
    store_example_confidence()




main()