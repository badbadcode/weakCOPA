from os.path import dirname, realpath, join
from pathlib import Path


class Config:

    EPOCH = 20
    PATIENCE = 5
    CKPT_DIR = "ckpt_20runs"
    RES_DIR = "result_20runs"

    BASE_DIR = Path(realpath(join(dirname(realpath(__file__)), "..")))
    DATA_DIR = Path(join(BASE_DIR, "weakCOPA/data"))

    train_data_fp = {"0": f"{DATA_DIR}/train.csv",
                     "bcopa": f"{DATA_DIR}/train_bcopa.csv"}

    dev_data_fp = {"0": f"{DATA_DIR}/dev.csv",
                     "bcopa": f"{DATA_DIR}/dev_bcopa.csv"}

    train_reverse_fp = "data/train_reverse.csv"
    dev_reverse_fp = "data/dev_reverse.csv"

    train_fp = {"0": "", "bcopa": "_Bcopa"}

    best_lr_dic_noadv = {"0": {"b-l": 1e-4, "rb-l": 8e-6, "db-l": 5e-6, "alb-xxl1": 1.1e-4},
                         "bcopa": {"b-l": 1e-4, "rb-l": 8e-6, "db-l": 5e-6, "alb-xxl1": 1.1e-4},
                         }

    best_lr_dic_adv = {"0": {"b-l": 8e-5, "rb-l": 1.2e-5, "db-l": 1e-5, "alb-xxl1": 6e-5},
                       "bcopa": {"b-l": 8e-5, "rb-l": 1.2e-5, "db-l": 1e-5, "alb-xxl1": 6e-5},
                       }

    best_lr_dic = {"adv": best_lr_dic_adv, "noadv": best_lr_dic_noadv}

    best_lamda_dic = {"b-l": 0.01,  # 0.009,
                      "alb-xxl1": 0.01,
                      "rb-l": 0.01,
                      "db-l": 0.01,
                      "b-l-nnsp": 0.01}

    RANDOM_SEEDS = [107, 117, 127, 137,
                    822, 832, 842, 852,
                    727, 737, 747, 757,
                    409, 419, 429, 439,
                    426, 436, 446, 456,
                    ]

    CUT_LST = ["b-l","rb-l","alb-xxl1","db-l"]
    ADV_LST = ["noadv","adv"]
    AUG_LST = ["0","bcopa"]

    FP_TEST_LST = [
        "data/dev.csv",
        'data/test.csv',
        'data/test_easy.csv',
        'data/test_hard.csv',
        'data/test_adv_rpwrong_copa.csv',  # 将错误答案替换成前提
        'data/test_adv_rprandom_copa.csv',  # 将错误答案随便替换成一个答案（从500个错误答案中随机选）
        'data/test_adv_blind_all_copa.csv',  # mask问题类型
        "data/test_ce&ce_reverse.csv"
    ]

    FP_TEST_3_LST = ['data/test_adv_rpwrong_copa.csv',
                      'data/test_adv_rprandom_copa.csv']


