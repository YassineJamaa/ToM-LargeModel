from src import Assessmentv4, LocImportantUnits
from benchmark import BenchmarkBaseline
from analysis.percentages.compute_accuracy import save_performances_top, save_performances_random
from analysis.percentages.compute_outliers import save_outliers_top, save_outliers_random

def assess_percentages(assess: Assessmentv4, benchmark: BenchmarkBaseline, model_name: str, percentages: list, loc: LocImportantUnits, nums: int=0):
    candidates = benchmark.data["cands_letter"].explode().unique().tolist()
    df_list = []
    add_no_lesion = True
    for percentage in percentages:
        mask = loc.get_masked_ktop(percentage)
        num_random = 0 if percentage == 0 else nums
        # Inference
        res = assess.experiment(benchmark, mask, no_lesion=add_no_lesion, num_random=num_random)
        if add_no_lesion:
            acc_no = (res["predict_no_ablation"] == res["answer_letter"]).mean()
            chance_threshold = res["answer_letter"].value_counts(normalize=True).max()
            add_no_lesion = False
            if acc_no < chance_threshold:
                raise ValueError(
                    f"Error: Accuracy {model_name} (%) = {acc_no:.2%} is below the chance threshold of {chance_threshold:.2%}"
                )
        df_list.append(res)
    save_performances_top(df_list, model_name, f"{type(benchmark).__name__}_{loc.model_name}", percentages)
    save_outliers_top(df_list, model_name, f"{type(benchmark).__name__}_{loc.model_name}", candidates, percentages)
    if nums > 0:
        save_performances_random(df_list, model_name, percentages)
        save_outliers_random(df_list, model_name, candidates, percentages)
    return 0