'''
Preprocess the COPA data to a sequence-classification problem.
'''

from xml.etree import ElementTree as et
import pandas as pd
import numpy as np
import random
import json

def xml2data(fp):
    text = open(fp, encoding='UTF-8').read()
    root = et.fromstring(text)
    items = root.findall('item')
    data = []
    for item in items:
        part = {}
        qhh = []
        for node in item:
            qhh.append(node.text)
        part['p_hh'] = qhh
        part['ans_id'] = item.get('most-plausible-alternative')
        part['q_type'] = item.get('asks-for')
        part['id'] = item.get("id")
        data.append(part)
    return data


def transform2bertinput(data):
    cause_0 = []
    effect_0 = []
    cause_1 = []
    effect_1 = []
    label = []
    for data_i in data:
        ans_id = int(data_i['ans_id']) - 1
        label.append(ans_id)

        premise = data_i['p_hh'][0]
        hyp_0 = data_i['p_hh'][1]
        hyp_1 = data_i['p_hh'][2]

        if data_i['q_type'] == 'cause':
            cause_0.append(hyp_0)
            effect_0.append(premise)
            cause_1.append(hyp_1)
            effect_1.append(premise)
        else:
            cause_0.append(premise)
            effect_0.append(hyp_0)
            cause_1.append(premise)
            effect_1.append(hyp_1)
    inputs = {'cause_0': cause_0, 'effect_0': effect_0, 'cause_1': cause_1, 'effect_1': effect_1, 'label': label}
    return inputs

def transform2bertinput_blind(data):
    cause_0 = []
    effect_0 = []
    cause_1 = []
    effect_1 = []
    label = []
    seed = 1124
    random.seed(seed)
    for data_i in data:
        ans_id = int(data_i['ans_id']) - 1
        label.append(ans_id)
        premise = data_i['p_hh'][0]
        hyp_0 = data_i['p_hh'][1]
        hyp_1 = data_i['p_hh'][2]
        a = random.randint(0, 1)
        if a == 0: # cause
            cause_0.append(hyp_0)
            effect_0.append(premise)
            cause_1.append(hyp_1)
            effect_1.append(premise)
        elif a == 1: #effect
            cause_0.append(premise)
            effect_0.append(hyp_0)
            cause_1.append(premise)
            effect_1.append(hyp_1)
    inputs = {'cause_0': cause_0, 'effect_0': effect_0, 'cause_1': cause_1, 'effect_1': effect_1, 'label': label}
    return inputs

def transform2bertinput_rpwrong(data):
    cause_0 = []
    effect_0 = []
    cause_1 = []
    effect_1 = []
    label = []
    for data_i in data:
        ans_id = int(data_i['ans_id']) - 1
        label.append(ans_id)

        premise = data_i['p_hh'][0]
        hyp_0 = data_i['p_hh'][1]
        hyp_1 = data_i['p_hh'][2]

        if data_i['q_type'] == 'cause':
            if ans_id == 0:
                cause_0.append(hyp_0)
                effect_0.append(premise)
                cause_1.append(premise)  # replacement
                effect_1.append(premise)
            else:
                cause_0.append(premise)  # replacement
                effect_0.append(premise)
                cause_1.append(hyp_1)
                effect_1.append(premise)
        else:
            if ans_id == 0:
                cause_0.append(premise)
                effect_0.append(hyp_0)
                cause_1.append(premise)
                effect_1.append(premise)  # replacement
            else:
                cause_0.append(premise)
                effect_0.append(premise) # replacement
                cause_1.append(premise)
                effect_1.append(hyp_1)

    inputs = {'cause_0': cause_0, 'effect_0': effect_0, 'cause_1': cause_1, 'effect_1': effect_1, 'label': label}
    return inputs

def transform2bertinput_rprandom(data):
    cause_0 = []
    effect_0 = []
    cause_1 = []
    effect_1 = []
    label = []

    wrong_alts =[]
    for data_i in data:
        ans_id = int(data_i['ans_id']) #1/2
        wrong_id = 3-ans_id #2/1
        wrong_alts.append(data_i['p_hh'][wrong_id])

    seed_val = 1124
    random.seed(seed_val)
    random_ans = random.sample(range(500),500)
    print(random_ans)
    for data_i,rand_ind in zip(data,random_ans):
        ans_id = int(data_i['ans_id']) - 1
        label.append(ans_id)

        premise = data_i['p_hh'][0]
        hyp_0 = data_i['p_hh'][1]
        hyp_1 = data_i['p_hh'][2]

        new_wrong = wrong_alts[rand_ind]

        if data_i['q_type'] == 'cause':
            if ans_id == 0:
                cause_0.append(hyp_0)
                effect_0.append(premise)
                cause_1.append(new_wrong)  # replacement
                effect_1.append(premise)
            else:
                cause_0.append(new_wrong)  # replacement
                effect_0.append(premise)
                cause_1.append(hyp_1)
                effect_1.append(premise)
        else:
            if ans_id == 0:
                cause_0.append(premise)
                effect_0.append(hyp_0)
                cause_1.append(premise)
                effect_1.append(new_wrong)  # replacement
            else:
                cause_0.append(premise)
                effect_0.append(new_wrong) # replacement
                cause_1.append(premise)
                effect_1.append(hyp_1)

    inputs = {'cause_0': cause_0, 'effect_0': effect_0, 'cause_1': cause_1, 'effect_1': effect_1, 'label': label}
    return inputs

def getTrainDev(data,data_bcopa):

    # Set the seed value all over the place to make this reproducible.
    seed_val = 610
    random.seed(seed_val)
    np.random.seed(seed_val)

    #copa
    dev_index = random.sample(range(500), 50)
    data_dev = [data[i] for i in sorted(dev_index)]
    data_train = [data[i] for i in range(500) if i not in sorted(dev_index)]

    #bcopa
    dev_index_bcopa = [2*x for x in dev_index]
    dev_bal = [2 * x+1 for x in dev_index]

    dev_index_bcopa.extend(dev_bal)
    print('dev_index',dev_index)
    print('dev_index_bcopa', dev_index_bcopa)
    print('sorted_dev_index_bcopa', sorted(dev_index_bcopa))
    data_dev_bcopa = [data_bcopa[i] for i in sorted(dev_index_bcopa)]
    data_train_bcopa = [data_bcopa[i] for i in range(1000) if i not in sorted(dev_index_bcopa)]

    inputs_train = transform2bertinput(data_train)
    inputs_dev = transform2bertinput(data_dev)

    inputs_train_bcopa = transform2bertinput(data_train_bcopa)
    inputs_dev_bcopa = transform2bertinput(data_dev_bcopa)

    return inputs_train, inputs_dev, inputs_train_bcopa, inputs_dev_bcopa


def getTestEasyHard(data_test):
    f = open("data/easy_hard_subsets.json", encoding='utf-8')
    setting = json.load(f)
    easy_index = setting['easy']
    hard_index = setting['hard']
    data_test_easy = [data_test[i-501] for i in easy_index]
    data_test_hard = [data_test[i-501] for i in hard_index]
    return data_test_easy, data_test_hard

def getReverseSave(fp_in,fp_out):
    df = pd.read_csv(fp_in)
    cause_0 = df["cause_0"].tolist()
    cause_1 = df["cause_1"].tolist()
    effect_0 = df["effect_0"].tolist()
    effect_1 = df["effect_1"].tolist()

    df["cause_0"] = effect_0
    df["effect_0"] = cause_0
    df["cause_1"] = effect_1
    df["effect_1"] = cause_1
    df.to_csv(fp_out, index=False, sep=',')

def write_csv(inputs, fp):
    df = pd.DataFrame(inputs)
    df.to_csv(fp, index=True, sep=',')
    print(f"{fp} is saved")




# read and split train/dev/test FROM COPA and BCOPA
fp_dev = 'data/copa-dev.xml'
fp_bcopa_train = 'data/balacopa-dev-all.xml'
fp_test = 'data/copa-test.xml'
data_test = xml2data(fp_test)
data = xml2data(fp_dev)
data_bcopa = xml2data(fp_bcopa_train)
inputs_train, inputs_dev, inputs_train_bcopa, inputs_dev_bcopa = getTrainDev(data, data_bcopa)
write_csv(inputs_train, 'data/train.csv')
write_csv(inputs_dev, 'data/dev.csv')
write_csv(inputs_train_bcopa, 'data/train_bcopa.csv')
write_csv(inputs_dev_bcopa,  'data/dev_bcopa.csv')
write_csv(transform2bertinput(data_test), 'data/test.csv')


# read test-easy, test-hard
data_test_easy, data_test_hard = getTestEasyHard(data_test)
write_csv(transform2bertinput(data_test_easy), 'data/test_easy.csv')
write_csv(transform2bertinput(data_test_hard), 'data/test_hard.csv')



# Make adv Test by replacing the correct answer with premise.
inputs_test_rpwrong = transform2bertinput_rpwrong(data_test)
file_path_test_rpwrong = 'data/test_adv_rpwrong_copa.csv'
write_csv(inputs_test_rpwrong, file_path_test_rpwrong)

# Make adv Test by replacing the correct answer with random wrong answer.
inputs_test_rprandom = transform2bertinput_rprandom(data_test)
file_path_test_rprandom = 'data/test_adv_rprandom_copa.csv'
write_csv(inputs_test_rprandom, file_path_test_rprandom)

# Masking the question type by giving a random "ask-for".
inputs_test_blind = transform2bertinput_blind(data_test)
file_path_test_blind = 'data/test_adv_blind_all_copa.csv'
write_csv(inputs_test_blind, file_path_test_blind)

# Save the input that the model will have If we ask the opposite question for the same instance in the original set
getReverseSave('data/train.csv', 'data/train_reverse.csv')
getReverseSave('data/dev.csv', 'data/dev_reverse.csv')

