import torch
from torch import nn
from transformers import BertForSequenceClassification, RobertaForSequenceClassification,AlbertForSequenceClassification, DebertaForSequenceClassification


def getDevice():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('using the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU , use the CPU instead.')
        device = torch.device("cpu")
    return device

class BertCOPA(nn.Module):
    def __init__(self, model_name):
        super(BertCOPA, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.concat = nn.Linear(self.bert.config.hidden_size, 1)
        self.device = getDevice()
        self.batch_size = 32
        self.weight_decay = 0.001
        self.warm_up = 0.1

    def forward(self, input_ids_0, input_ids_1, input_mask_0, input_mask_1, token_type_ids=None, labels=None):
        outputs_0 = self.bert(input_ids_0,
                             token_type_ids=None,
                             attention_mask= input_mask_0, output_hidden_states=True)
        outputs_1 = self.bert(input_ids_1,
                             token_type_ids=None,
                             attention_mask= input_mask_1, output_hidden_states=True)

        z0_0 = outputs_0.hidden_states[-1][:, 0, :]  # the first hidden_states of the first choice
        z0_1 = outputs_1.hidden_states[-1][:, 0, :]  # the first hidden_states of the second choice

        y0 = self.concat(z0_0)
        y1 = self.concat(z0_1)

        logits = torch.cat((y0, y1), -1)
        probs = nn.functional.softmax(logits,dim=1)
        loss = nn.functional.cross_entropy(probs, labels)

        loss_adv = torch.mean(nn.functional.pairwise_distance(y0, y1))
        return logits,probs,loss,loss_adv


class RobertaCOPA(nn.Module):
    def __init__(self, model_name):
        super(RobertaCOPA, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # self.bert = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path="/nfsshare/home/hanmingyue/whl/roberta.large/", from_pt = True,num_labels=2)
        self.concat = nn.Linear(self.bert.config.hidden_size, 1)
        self.device = getDevice()
        self.batch_size = 32
        self.weight_decay = 0.01
        self.warm_up = 0.06


    def forward(self, input_ids_0, input_ids_1, input_mask_0, input_mask_1, token_type_ids=None, labels=None):
        outputs_0 = self.bert(input_ids_0,
                             token_type_ids=None,
                             attention_mask= input_mask_0, output_hidden_states=True)
        outputs_1  = self.bert(input_ids_1,
                             token_type_ids=None,
                             attention_mask= input_mask_1, output_hidden_states=True)



        z0_0 = outputs_0.hidden_states[-1][:, 0, :]  # the first hidden_states of the first choice
        z0_1 = outputs_1.hidden_states[-1][:, 0, :]  # the first hidden_states of the second choice


        y0 = self.concat(z0_0)
        y1 = self.concat(z0_1)

        logits = torch.cat((y0, y1), -1)
        probs = nn.functional.softmax(logits,dim=1)
        loss = torch.nn.functional.cross_entropy(probs, labels)

        loss_adv = torch.mean(nn.functional.pairwise_distance(y0, y1))
        return logits,probs,loss,loss_adv

class AlbertCOPA(nn.Module):
    def __init__(self, model_name):
        super(AlbertCOPA, self).__init__()
        self.bert = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.concat = nn.Linear(self.bert.config.hidden_size, 1)
        self.device = getDevice()
        self.batch_size = 32
        self.weight_decay = 0
        self.warm_up = 0


    def forward(self, input_ids_0, input_ids_1, input_mask_0, input_mask_1, token_type_ids=None, labels=None):
        outputs_0 = self.bert(input_ids_0,
                             token_type_ids=None,
                             attention_mask= input_mask_0, output_hidden_states=True)
        outputs_1  = self.bert(input_ids_1,
                             token_type_ids=None,
                             attention_mask= input_mask_1, output_hidden_states=True)

        z0_0 = outputs_0.hidden_states[-1][:, 0, :]  # the first hidden_states of the first choice
        z0_1 = outputs_1.hidden_states[-1][:, 0, :]  # the first hidden_states of the second choice
        y0 = self.concat(z0_0)
        y1 = self.concat(z0_1)
        logits = torch.cat((y0, y1), -1)
        probs = nn.functional.softmax(logits,dim=1)
        loss = torch.nn.functional.cross_entropy(probs, labels)
        loss_adv = torch.mean(nn.functional.pairwise_distance(y0, y1))
        return logits,probs,loss,loss_adv

class DeBertaCOPA(nn.Module):
    def __init__(self, model_name):
        super(DeBertaCOPA, self).__init__()
        # self.bert = DebertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path="/nfsshare/home/hanmingyue/whl/deberta-large", num_labels=2)
        self.bert = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.concat = nn.Linear(self.bert.config.hidden_size, 1)
        self.device = getDevice()
        self.batch_size = 32
        self.weight_decay = 0.01
        self.warm_up = 0.06

    def forward(self, input_ids_0, input_ids_1, input_mask_0, input_mask_1, token_type_ids=None, labels=None):
        outputs_0 = self.bert(input_ids_0,
                             token_type_ids=None,
                             attention_mask= input_mask_0, output_hidden_states=True)
        outputs_1  = self.bert(input_ids_1,
                             token_type_ids=None,
                             attention_mask= input_mask_1, output_hidden_states=True)

        z0_0 = outputs_0.hidden_states[-1][:, 0, :]  # the first hidden_states of the first choice
        z0_1 = outputs_1.hidden_states[-1][:, 0, :]  # the first hidden_states of the second choice
        y0 = self.concat(z0_0)
        y1 = self.concat(z0_1)
        logits = torch.cat((y0, y1), -1)
        probs = nn.functional.softmax(logits,dim=1)
        loss = torch.nn.functional.cross_entropy(probs, labels)

        loss_adv = torch.mean(nn.functional.pairwise_distance(y0, y1))
        return logits,probs,loss,loss_adv