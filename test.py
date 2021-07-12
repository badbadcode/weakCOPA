
import torch
import pandas as pd
from tqdm import tqdm
from config import Config
from utils import getSaveModelPath, getModel, get3acc
from utils import getTestLoader, getACC, write_add_csv

# CUT_LST = ["b-l","rb-l","alb-xxl1","db-l"]
# ADV_LST = ["noadv","adv"]
# AUG_LST = ["0","bcopa"]
# RANDOM_SEEDS = [107, 117, 127, 137,
#                 822, 832, 842, 852,
#                 727, 737, 747, 757,
#                 409, 419, 429, 439,
#                 426, 436, 446, 456,
#                 ]

for model_shortcut in Config.CUT_LST[3:]:
    for IsAdv in Config.ADV_LST[:]:
        for aug_data in Config.AUG_LST[:1]:
            best_lr = Config.best_lr_dic[IsAdv][aug_data][model_shortcut]  # selected by pfm of dev set before
            lamda = Config.best_lamda_dic[model_shortcut]
            model, model_name = getModel(model_shortcut)
            res_save_path = f"{Config.RES_DIR}/testing_20runs.csv"
            data_test_fp = 'data/test.csv'
            copa_test_loader = getTestLoader(model, data_test_fp, model_name)
            for seed in tqdm(Config.RANDOM_SEEDS[17:18]):
                res_dic = {"model_shortcut":model_shortcut,
                           "IsAdv":IsAdv,
                           "aug_data":aug_data,
                           "seed":seed}
                for fp_test in tqdm(Config.FP_TEST_LST):
                    data_fp = fp_test.split(".")[0].split("/")[-1]  # "dev"
                    print(
                        f"============= Test on {fp_test} ============= with {model_name} ============ best lr: {best_lr} ======================")

                    df = pd.read_csv(fp_test, encoding='latin-1', header=0)
                    save_model_path = getSaveModelPath(model_shortcut, aug_data, seed, IsAdv, best_lr)
                    print(f"Loading {save_model_path} Now.....")
                    model.load_state_dict(torch.load(save_model_path))
                    test_loader = getTestLoader(model, fp_test, model_name)
                    if fp_test in Config.FP_TEST_3_LST:
                        _, _, true_labels, probs = getACC(model, test_loader)
                        _, _, true_labels, probs_copa_test = getACC(model, copa_test_loader)
                        acc = get3acc(probs, probs_copa_test, true_labels)
                    else:
                        acc, pred_flat, labels_flat, prob_flat = getACC(model, test_loader)
                    res_dic[data_fp] = acc
                    print("seed:", seed, f"-----{fp_test} Choice Accuracy on Best Model: {acc}")
                write_add_csv(res_dic, res_save_path)