import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import sys
from earlystopping import EarlyStopping
from utils import getACC, write_add_csv
from utils import SetupSeed, getModel, getTrainDevLoader, getSaveModelPath, getOutputs, getTrainDevLoader_ADV
from config import Config
import os
import random

# model_shortcut = sys.argv[1]  # ["b-l", "rb-l", "db-l", "alb-xxl1"]
# seed_val = int(sys.argv[2])
# isadv = sys.argv[3]  # ["adv", "noadv"]
# aug_data = sys.argv[4]  # ["0", "bcopa"]


model_shortcut = "db-l"  # ["b-l", "rb-l", "db-l", "alb-xxl1"]
seed_val = 436
isadv = "adv"  # ["adv", "noadv"]
aug_data = "0"  # ["0", "bcopa"]

lr = Config.best_lr_dic[isadv][aug_data][model_shortcut]  # selected by pfm of dev set before
lamda = Config.best_lamda_dic[model_shortcut]
epochs = Config.EPOCH
patience = Config.PATIENCE

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True


train_data_fp = Config.train_data_fp[aug_data]
dev_data_fp = Config.dev_data_fp[aug_data]
train_data_reverse_fp = Config.train_reverse_fp
dev_data_reverse_fp = Config.dev_reverse_fp


# SetupSeed(seed_val)
save_model_path = getSaveModelPath(model_shortcut, aug_data, seed_val, isadv, lr)
model, model_name = getModel(model_shortcut)
if model.device == torch.device("cuda"):
    model.cuda()

if isadv == "noadv":
    train_dataloader, validation_dataloader = getTrainDevLoader(model_name, train_data_fp, dev_data_fp, model.batch_size)
elif isadv =="adv":
    train_dataloader, validation_dataloader = getTrainDevLoader_ADV(model_name, train_data_fp, train_data_reverse_fp, dev_data_fp,dev_data_reverse_fp, model.batch_size)



optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=1e-8,
                  weight_decay=model.weight_decay,
                  )

total_steps = len(train_dataloader) * epochs  # [number of batches] x [number of epochs].
print("total_steps", total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(model.warm_up * total_steps),
                                            # Default value in run_glue.py
                                            num_training_steps=total_steps)

early_stopping = EarlyStopping(patience=patience, verbose=True,
                               path=save_model_path)  # if valid_loss didn't improve for patience epochs, we stop and save the best one.



for epoch_i in range(0, epochs):

    train_losses = []
    dev_losses = []
    # ========================================
    #               Training
    # ========================================
    print("")
    print('================ Epoch {:} / {:} ================='.format(epoch_i + 1, epochs))
    print("model_name--", model_name, "  lr--", lr, "  random_seed--", seed_val)
    print(f'=========={model_name} Training ==========')

    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        if isadv == "noadv":
            logits, probs, loss, _ = getOutputs(model, batch)
        elif isadv == "adv":
            batch_ori = batch[:5]
            logits, probs, loss, _ = getOutputs(model, batch_ori)
            batch_rev = batch[5:]
            _, _, _, loss_adv = getOutputs(model, batch_rev)
            loss = (1 - lamda) * loss + lamda * loss_adv

        if step % 4 == 0:
            print('step loss:', loss)
        loss.backward()
        if model_shortcut == 'b-l':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

    # EARLY STOPPING
    ######################
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    for step, dev_batch in enumerate(validation_dataloader):
        if isadv == "noadv":
            dev_logits, dev_probs, dev_loss, _ = getOutputs(model, dev_batch)
        elif isadv == "adv":
            batch_ori = dev_batch[:5]
            logits, probs, loss_ori, _ = getOutputs(model, batch_ori)
            batch_rev = batch[5:]
            _, _, _, loss_adv = getOutputs(model, batch_rev)
            dev_loss = (1 - lamda) * loss_ori + lamda * loss_adv

        dev_losses.append(dev_loss.item())
    # calculate average loss over an epoch (all batches)
    train_loss = np.average(train_losses)
    dev_loss = np.average(dev_losses)
    epoch_len = len(str(epochs))
    print_msg = (f'[{epoch_i + 1:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'dev_loss: {dev_loss:.5f}')
    print(print_msg)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation and train set and save the best model so far.
    acc_train,_,_,_ = getACC(model, train_dataloader)
    acc_dev,_,_,_ = getACC(model, validation_dataloader)
    print("Train Choice Accuracy on Best Model so far: {0:.4f}".format(acc_train))
    print("Dev Choice Accuracy on Best Model so far: {0:.4f}".format(acc_dev))
    early_stopping(dev_loss, model)
    best_epoch = epochs - early_stopping.counter
    if early_stopping.early_stop:
        print(f"Early stopping {model_name}")
        best_epoch = epoch_i + 1 - patience
        break
print("Training complete!")


print("========================Save the pfm of dev set on the Best Model=====================")
print("model_name--", model_shortcut, "  lr--", lr, "  random_seed--", seed_val)
print("Best epoch is :", best_epoch)
# load the last checkpoint with the best model and evaluate on all dataset
model.load_state_dict(torch.load(save_model_path))
acc_train,_,_,_ = getACC(model, train_dataloader)
acc_dev,_,_,_ = getACC(model, validation_dataloader)
print("Train Choice Accuracy on Best Model so far: {0:.4f}".format(acc_train))
print("Dev Choice Accuracy on Best Model so far: {0:.4f}".format(acc_dev))
res_dict = {"model_name": model_shortcut,
            "lr": lr,
            "isadv": isadv,
            "aug_data":aug_data,
            "random_seed": seed_val,
            "best_epoch": best_epoch,
            "patience": patience,
            'acc_train': acc_train,
            "acc_dev": acc_dev}
write_add_csv(res_dict, f"{Config.RES_DIR}/dev_20runs.csv")
