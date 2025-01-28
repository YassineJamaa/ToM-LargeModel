import numpy as np
import pandas as pd
from analysis.script import compute_t_confidence_interval
from analysis.percentages.utils import save_result_json

def compute_top_diff(res: pd.DataFrame):
    # Compute Top-K%
    acc_top = (res["predict_ablate_top"] == res["answer_letter"]).mean()
    return acc_top

def compute_random_diff(res: pd.DataFrame):
    cols = res.filter(like='predict_ablate_random').columns
    if cols.empty:
        return None, None
    # Compute Random-K%
    mean_random_array = np.array([
        (res[col] == res['answer_letter']).mean() 
        for col in cols
    ])
    acc_random, lower_random, _ = compute_t_confidence_interval(mean_random_array)
    return acc_random, lower_random
    

def save_performances_random(df_list: list, model_name: str, percentages: list):
    rand_diffs = []
    rand_errs = []
    res0 = df_list[0]
    acc_no = (res0["predict_no_ablation"] == res0["answer_letter"]).mean()
    chance_threshold = res0["answer_letter"].value_counts(normalize=True).max()
    for df in df_list:
        rand_acc, rand_lower = compute_random_diff(df)
        if rand_acc is None:
            rand_diffs.append(0)
            rand_errs.append(0)
        else:
            rand_diffs.append((rand_acc-acc_no)*100)
            rand_errs.append((rand_acc-rand_lower)*100)
    data_json = {
        "percentages": [percentage*100 for percentage in percentages],
        "rand_diffs": rand_diffs,
        "rand_errs": rand_errs,
        "acc_no": acc_no,
        "chance_threshold": chance_threshold
        }
    save_result_json(data_json, model_name, "random", "diff_acc_pct")

def save_performances_top(df_list: list, model_name: str, loc_name: str, percentages: list):
    acc_diffs = []
    res0 = df_list[0]
    acc_no = (res0["predict_no_ablation"] == res0["answer_letter"]).mean()
    chance_threshold = res0["answer_letter"].value_counts(normalize=True).max()
    for df in df_list:
        acc_top = compute_top_diff(df)
        acc_diffs.append((acc_top-acc_no)*100)
    data_json = {
        "percentages": [percentage*100 for percentage in percentages],
        "acc_diffs": acc_diffs,
        "acc_no": acc_no,
        "chance_threshold": chance_threshold
        }
    save_result_json(data_json, model_name, loc_name, "diff_acc_pct")

