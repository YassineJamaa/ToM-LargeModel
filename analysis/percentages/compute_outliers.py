import numpy as np
import pandas as pd
from analysis.script import compute_t_confidence_interval
from analysis.percentages.utils import save_result_json

def compute_top_outliers(res: pd.DataFrame, candidates: list):
    outlier_pred_top = np.array([item for item in res["predict_ablate_top"].tolist() if item not in candidates])
    return len(outlier_pred_top), np.unique(outlier_pred_top.flatten()).tolist()

def compute_random_outliers(res: pd.DataFrame, candidates: list):
    cols = res.filter(like='predict_ablate_random').columns
    if cols.empty:
        return 0, 0, []
    outlier_randoms = []
    outliers_list = []
    for idx, col in enumerate(cols):
        outliers = np.array([item for item in res[col].tolist() if item not in candidates])
        outlier_randoms.append(len(outliers))
        outliers_list.append(np.unique(outliers).tolist())
    outlier_randoms_array = np.array(outlier_randoms)
    outlier_random, lower_random, _ = compute_t_confidence_interval(outlier_randoms_array)
    return outlier_random, lower_random, outliers_list

def save_outliers_top(df_list: list, model_name: str, loc_name: str, candidates: list, percentages: list):
    top_outliers = []
    outliers_list = []
    for df in df_list:
        n_outliers, outliers = compute_top_outliers(df, candidates)
        outliers_list.append(outliers)
        top_outliers.append(n_outliers)
    data_json = {
        "percentages": [percentage*100 for percentage in percentages],
        "top_outliers": top_outliers,
        "outliers": outliers_list
    }
    save_result_json(data_json, model_name, loc_name, "outliers")

def save_outliers_random(df_list: list, model_name: str, candidates: list, percentages: list):
    rand_outliers = []
    rand_errs = []
    outliers_list = []
    for df in df_list:
        mean_outliers, lower_outliers, outliers = compute_random_outliers(df, candidates)
        rand_outliers.append(mean_outliers)
        rand_errs.append(mean_outliers - lower_outliers)
        outliers_list.append(outliers)
    data_json = {
        "percentages": [percentage*100 for percentage in percentages],
        "rand_outliers": rand_outliers,
        "rand_errs": rand_errs,
        "outliers": outliers_list
    }
    save_result_json(data_json, model_name, "random", "outliers")