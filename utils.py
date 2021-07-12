'''
Funtions in many jobs.
'''
from models import BertCOPA, RobertaCOPA, AlbertCOPA, DeBertaCOPA
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import os
import random
from config import Config

def write_add_csv(inputs, fp):
    df = pd.DataFrame(inputs, index=[0])
    dir = "/".join(fp.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(fp):
        df.to_csv(fp, index=True, sep=',', header=True)
        print(f"{fp} is saved first time")
    else:
        df.to_csv(fp, mode='a', index=True, sep=',', header=False)
        print(f"{fp} added the new data")

def SetupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def getModel(model_shortcut):
    if model_shortcut in ['b-b']:
        model_name = 'bert-base-uncased'
        model = BertCOPA(model_name)
    elif model_shortcut in ['b-l']:
        model_name = 'bert-large-uncased'
        model = BertCOPA(model_name)
    elif model_shortcut in ['rb-l']:
        model_name = 'roberta-large'
        model = RobertaCOPA(model_name)
    elif model_shortcut in ['alb-xxl1']:
        model_name = 'albert-xxlarge-v1'
        model = AlbertCOPA(model_name)
    elif model_shortcut in ['db-l']:
        model_name = 'microsoft/deberta-large'
        model = DeBertaCOPA(model_name)
    return model, model_name


def getSaveModelPath(model_shortcut, aug_data, seed_val, isadv, lr):
    train_fp = Config.train_fp[aug_data]  # for saving the ckpt, {"0": "", "bcopa": "_Bcopa"}
    ckpt_path = f"{Config.CKPT_DIR}/ckpt_{model_shortcut}{train_fp}_{isadv}/"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_model_path = f'{ckpt_path}checkpoint_{str(seed_val)}_lr_{str(lr)}.pt'
    return save_model_path

def getDevResPath(train_data):
    res_path = f"{Config.RES_DIR}/{train_data}/{Config.csv_dev_pfm}"
    return res_path

def getOutputs(model, batch):
    b_input_ids_0 = batch[0].to(model.device)
    # print(b_input_ids_0[0])
    b_input_mask_0 = batch[1].to(model.device)
    b_input_ids_1 = batch[2].to(model.device)
    b_input_mask_1 = batch[3].to(model.device)
    b_labels = batch[4].to(model.device)
    logits, probs, loss, loss_adv = model(b_input_ids_0, b_input_ids_1, b_input_mask_0, b_input_mask_1, labels=b_labels)
    return logits, probs, loss, loss_adv


def getInput(filename, model_name):
    df = pd.read_csv(filename, encoding='latin-1', header=0)
    causes_0 = df['cause_0'].tolist()
    effects_0 = df['effect_0'].tolist()
    causes_1 = df['cause_1'].tolist()
    effects_1 = df['effect_1'].tolist()
    labels = df['label'].tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    encoded_inputs_0 = tokenizer(causes_0, effects_0, padding='max_length', truncation=True, max_length=32)
    encoded_inputs_1 = tokenizer(causes_1, effects_1, padding='max_length', truncation=True, max_length=32)
    input_ids_0 = encoded_inputs_0["input_ids"]
    input_ids_1 = encoded_inputs_1["input_ids"]
    attention_masks_0 = encoded_inputs_0["attention_mask"]
    attention_masks_1 = encoded_inputs_1["attention_mask"]
    return input_ids_0, attention_masks_0, input_ids_1, attention_masks_1, labels

def inputs2tensor(inputs_0, masks_0, inputs_1, masks_1, labels):
    inputs_0 = torch.tensor(inputs_0)
    masks_0 = torch.tensor(masks_0)
    inputs_1 = torch.tensor(inputs_1)
    masks_1 = torch.tensor(masks_1)
    labels = torch.tensor(labels)
    tensor_dataset = TensorDataset(inputs_0, masks_0, inputs_1, masks_1, labels)
    return tensor_dataset

def inputs2adv_tensor(inputs_0, masks_0, inputs_1, masks_1,train_reverse_inputs_0,
                               train_reverse_masks_0, train_reverse_inputs_1, train_reverse_masks_1, labels):
    inputs_0 = torch.tensor(inputs_0)
    masks_0 = torch.tensor(masks_0)
    inputs_1 = torch.tensor(inputs_1)

    masks_1 = torch.tensor(masks_1)
    train_reverse_inputs_0 = torch.tensor(train_reverse_inputs_0)
    train_reverse_masks_0 = torch.tensor(train_reverse_masks_0)
    train_reverse_inputs_1 = torch.tensor(train_reverse_inputs_1)
    train_reverse_masks_1 = torch.tensor(train_reverse_masks_1)
    labels = torch.tensor(labels)

    tensor_dataset = TensorDataset(inputs_0, masks_0, inputs_1, masks_1, labels,
                                   train_reverse_inputs_0, train_reverse_masks_0,
                                   train_reverse_inputs_1, train_reverse_masks_1, labels)
    return tensor_dataset

# From csv to model_input, to tensor_dataset, to data_loader  --- train,dev,test
def getTrainDevLoader(model_name, train_fp, dev_fp, batch_size):
    train_inputs_0, train_masks_0, train_inputs_1, train_masks_1, train_labels = getInput(train_fp, model_name)
    validation_inputs_0, validation_masks_0, validation_inputs_1, validation_masks_1, validation_labels = getInput(
        dev_fp, model_name)
    train_data = inputs2tensor(train_inputs_0, train_masks_0, train_inputs_1, train_masks_1, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = inputs2tensor(validation_inputs_0, validation_masks_0, validation_inputs_1, validation_masks_1,
                                    validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader

def getTrainDevLoader_ADV(model_name, train_fp, train_reverse_fp, dev_fp,dev_reverse_fp, batch_size):
    train_inputs_0, train_masks_0, train_inputs_1, train_masks_1, train_labels = getInput(train_fp, model_name)
    train_reverse_inputs_0, train_reverse_masks_0, train_reverse_inputs_1, train_reverse_masks_1, train_reverse_labels = getInput(
        train_reverse_fp, model_name)

    validation_inputs_0, validation_masks_0, validation_inputs_1, validation_masks_1, validation_labels = getInput(
        dev_fp, model_name)
    validation_reverse_inputs_0, validation_reverse_masks_0, validation_reverse_inputs_1, validation_reverse_masks_1, validation_reverse_labels = getInput(
        dev_reverse_fp, model_name)

    train_data = inputs2adv_tensor(train_inputs_0, train_masks_0, train_inputs_1, train_masks_1, train_reverse_inputs_0,
                                   train_reverse_masks_0, train_reverse_inputs_1, train_reverse_masks_1, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = inputs2adv_tensor(validation_inputs_0, validation_masks_0, validation_inputs_1,
                                        validation_masks_1, validation_reverse_inputs_0, validation_reverse_masks_0,
                                        validation_reverse_inputs_1, validation_reverse_masks_1,
                                        validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


# From csv to model_input, to tensor_dataset, to data_loader  --- train,dev,test
def getTestLoader(model,fp, model_name):
    test_inputs_0, test_masks_0, test_inputs_1, test_masks_1, test_labels = getInput(fp, model_name)
    test_data = inputs2tensor(test_inputs_0, test_masks_0, test_inputs_1, test_masks_1, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=model.batch_size)
    return test_dataloader


def getACC(model, prediction_dataloader):
    if model.device == torch.device("cuda"):
        model.cuda()
    model.eval()
    logits, predictions, true_labels, probs = [], [], [], []
    for batch in prediction_dataloader:
        batch = tuple(t.to(model.device) for t in batch[:5])
        b_input_ids_0, b_input_mask_0, b_input_ids_1, b_input_mask_1,b_labels = batch
        with torch.no_grad():
            logit, prob, _, _ = model(b_input_ids_0, b_input_ids_1, b_input_mask_0, b_input_mask_1, labels=b_labels)
        true_labels.append(b_labels)
        probs.append(prob)

    true_labels = [x.tolist() for x in true_labels]
    probs = [x.tolist() for x in probs]

    prob_flat = sum(probs, [])
    labels_flat = sum(true_labels, [])
    pred_flat = np.argmax(prob_flat, axis=1)
    acc = accuracy_score(labels_flat, pred_flat)

    return acc, pred_flat, labels_flat, prob_flat


def get3acc(probs, probs_copa_test, true_labels):
    acc_counts = 0
    for prob, prob_copa_test, label in zip(probs, probs_copa_test, true_labels):
        test_ans = np.argmax(prob_copa_test)
        adv_ans = np.argmax(prob)
        if test_ans == adv_ans:
            if test_ans == label:
                acc_counts += 1
    acc = acc_counts / 500.00  # 3个选项的都是500个test instances
    return acc


