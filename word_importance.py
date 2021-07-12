# case study
#imortance is calculated referring by
'''
Jiwei Li, Will Monroe, and Dan Jurafsky. Understanding Neural Networks through Representation Erasure.
'''
import torch
from transformers import AutoTokenizer
from utils import inputs2tensor,getACC
from torch.utils.data import DataLoader, SequentialSampler
import math
import pandas as pd
from nltk import word_tokenize
from models import DeBertaCOPA


def mask_exp(words,i):
    new_words = []
    for j in range(len(words)):
        if j !=i:
            new_words.append(words[j])
        else:
            new_words.append("[MASK]")
    return new_words


def get_log_label(premise, hyp0, hyp1, q_type, label,model_name,model):
    if q_type == "cause":
        causes_0 = [hyp0]
        effects_0 = [premise]
        causes_1 = [hyp1]
        effects_1 = [premise]
    elif q_type == "effect":
        causes_0 = [premise]
        effects_0 = [hyp0]
        causes_1 = [premise]
        effects_1 = [hyp1]
    test_labels = [label]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    encoded_inputs_0 = tokenizer(causes_0, effects_0, padding='max_length', truncation=True, max_length=32)
    encoded_inputs_1 = tokenizer(causes_1, effects_1, padding='max_length', truncation=True, max_length=32)
    test_inputs_0 = encoded_inputs_0["input_ids"]
    test_inputs_1 = encoded_inputs_1["input_ids"]
    test_masks_0 = encoded_inputs_0["attention_mask"]
    test_masks_1 = encoded_inputs_1["attention_mask"]
    test_data = inputs2tensor(test_inputs_0, test_masks_0, test_inputs_1, test_masks_1, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
    _, _, _, prob_flat = getACC(model, test_loader)
    prob_pred = prob_flat[0][label]
    negative_log_label = -1*math.log(prob_pred)
    return negative_log_label

def get_premise_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name, model):
    premise_words = word_tokenize(premise)
    score_list = []
    for i in range(len(premise_words)):
        new_words = mask_exp(premise_words,i)
        log_label_mask = get_log_label(" ".join(new_words),hyp0,hyp1, q_type, label,model_name, model)
        print(new_words, "log_label:log_label_mask",log_label,log_label_mask)
        score = (log_label_mask-log_label) / log_label
        score_list.append(score)
    return score_list

def get_h0_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name,model):
    hyp0 = word_tokenize(hyp0)
    score_list = []
    for i in range(len(hyp0)):
        new_words = mask_exp(hyp0, i)
        log_label_mask = get_log_label(premise, " ".join(new_words), hyp1, q_type, label, model_name,  model)
        print(new_words, "log_label:log_label_mask",log_label,log_label_mask)
        score = (log_label_mask-log_label) / log_label
        score_list.append(score)
    return score_list

def get_h1_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name, model):
    hyp1 = word_tokenize(hyp1)
    score_list = []
    for i in range(len(hyp1)):
        new_words = mask_exp(hyp1, i)
        log_label_mask = get_log_label(premise, hyp0, " ".join(new_words),q_type, label, model_name, model)
        print(new_words, "log_label:log_label_mask",log_label,log_label_mask)
        score = (log_label_mask-log_label) / log_label
        score_list.append(score)
    return score_list

def write_score(premise,hyp0,hyp1,instance_score,path):
    with open(path,"w") as f:
        f.writelines(instance_score["p"])
        f.writelines(word_tokenize(premise))
        f.writelines(instance_score["h0"])
        f.writelines(word_tokenize(hyp0))
        f.writelines(instance_score["h1"])
        f.writelines(word_tokenize(hyp1))
        f.close()

def pipeline(instance,model_path):
    premise = instance["p"]
    hyp0 = instance["h0"]
    hyp1 = instance["h1"]
    q_type = instance["q_type"]
    label = int(instance["label"])

    # IsAdv = model_path.split("/")[-2].split("-")[-1]
    model_name = 'microsoft/deberta-large'

    model = DeBertaCOPA(model_name)
    model.load_state_dict(torch.load(model_path))

    log_label = get_log_label(premise, hyp0, hyp1, q_type, label,model_name, model)
    p_score_list = get_premise_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name, model)
    h0_score_list= get_h0_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name, model)
    h1_score_list = get_h1_scores(premise,hyp0,hyp1, q_type, label,log_label,model_name, model)

    instance_score = {"p":p_score_list,
                      "h0":h0_score_list,
                      "h1":h1_score_list}

    return instance_score

model_path_ori = r"ckpt_20runs/ckpt_db-l_noadv/checkpoint_436_lr_5e-06.pt"
model_path_adv = r"ckpt_20runs/ckpt_db-l_adv/checkpoint_436_lr_1e-05.pt"

instance = {"p": "The man's voice projected clearly throughout the auditorium.",
       "h0": "He greeted the audience.",
       "h1": "He spoke into the microphone.",
       "q_type": "cause",
       "label": 1}

words = []
for sen in [instance["p"],instance["h0"],instance["h1"]]:
    for w in word_tokenize(sen):
        words.append(w)

instance_score_ori = pipeline(instance,model_path_ori)
instance_score_adv = pipeline(instance,model_path_adv)

ori_scores = instance_score_ori["p"] + instance_score_ori["h0"] + instance_score_ori["h1"]

sen_names = ["p"]*len(instance_score_ori["p"]) + ["wrong"] * len(instance_score_ori["h0"]) + ["correct"] * len(instance_score_ori["h1"])
adv_scores = instance_score_adv["p"] +  instance_score_adv["h0"] + instance_score_adv["h1"]

data_dic = {"names":sen_names,"words":words, "db-l":ori_scores, "db-l-reg":adv_scores}

df = pd.DataFrame(data_dic)
path = "casestudy/word_importance.csv"
df.to_csv(path,  header=True)

print("instance_score_ori",instance_score_ori)
print("instance_score_adv",instance_score_adv)
