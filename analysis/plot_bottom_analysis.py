import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_bottom_analysis(model_checkpoint, folder, valid_elements, group_by=None):
    model_name = model_checkpoint.split("/")[-1]
    json_filename = os.path.join("Result", folder, model_name, "MATHBenchmark_bottom.json")
    
    with open(json_filename, "r") as file:
        data_dict = json.load(file)

    # benchmark & bot_predictions & top_predictions
    benchmark = pd.DataFrame(data_dict["benchmark"])
    bot_predictions = np.array(data_dict["bot_predictions"])
    top_predictions = np.array(data_dict["top_predictions"])
    percentages = np.linspace(0.0, 0.025, 26)[::2] * 100

    # Outliers
    #valid_elements = {'a', 'b', 'c', 'd'}

    if group_by in ["Type", "Level"]:
        for group_value, group in benchmark.groupby(group_by):
            indices = group.index

            # Accuracies
            bot_accs = (bot_predictions[:, indices] == group["answer_letter"].values).mean(axis=1)
            top_accs = (top_predictions[:, indices] == group["answer_letter"].values).mean(axis=1)
            chance_threshold = group["answer_letter"].value_counts(normalize=True).max() * 100
            no_lesion = bot_accs[0] * 100

            # Outliers
            bot_outliers_mask = ~np.isin(bot_predictions[:, indices], list(valid_elements))
            top_outliers_mask = ~np.isin(top_predictions[:, indices], list(valid_elements))
            bot_outliers = np.sum(bot_outliers_mask, axis=1)
            top_outliers = np.sum(top_outliers_mask, axis=1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

            # Plot ax1
            ax1.plot(percentages, bot_accs * 100, marker="o", label="Bot-Lesion", color="orange")
            ax1.plot(percentages, top_accs * 100, marker="o", label="Top-Lesion", color="skyblue")
            ax1.axhline(y=no_lesion, color="green", linestyle="--")
            ax1.axhline(y=chance_threshold, color="red", linestyle="--")

            # Plot ax2
            ax2.plot(percentages, bot_outliers, marker="o", label="Bot-Lesion", color="orange")
            ax2.plot(percentages, top_outliers, marker="o", label="Top-Lesion", color="skyblue")
            ax2.set_xlabel("Percentages (%)", fontsize=10, fontweight="bold")
            ax2.set_ylabel("Outliers (#)", fontsize=10, fontweight="bold")
            ax2.grid()

            # Labels and title ax1
            ax1.text(percentages[-3], no_lesion + 0.5, "No Lesion Acc.", color="green", fontsize=10, fontweight="bold")
            ax1.text(percentages[0], chance_threshold + 0.5, "Chance threshold", color="red", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Percentages (%)", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
            ax1.set_title(f"{model_name}\nAccuracy by Lesion Percentage for {group_by}: {group_value}", fontsize=12, fontweight="bold")
            ax1.legend()
            ax1.grid()

    else:
        bot_accs = (bot_predictions == benchmark["answer_letter"].values).mean(axis=1)
        top_accs = (top_predictions == benchmark["answer_letter"].values).mean(axis=1)
        chance_threshold = benchmark["answer_letter"].value_counts(normalize=True).max() * 100
        no_lesion = bot_accs[0] * 100

        # Outliers
        bot_outliers_mask = ~np.isin(bot_predictions, list(valid_elements))
        top_outliers_mask = ~np.isin(top_predictions, list(valid_elements))
        bot_outliers = np.sum(bot_outliers_mask, axis=1)
        top_outliers = np.sum(top_outliers_mask, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # Plot ax1
        ax1.plot(percentages, bot_accs * 100, marker="o", label="Bot-Lesion", color="orange")
        ax1.plot(percentages, top_accs * 100, marker="o", label="Top-Lesion", color="skyblue")
        ax1.axhline(y=no_lesion, color="green", linestyle="--")
        ax1.axhline(y=chance_threshold, color="red", linestyle="--")

        # Plot ax2
        ax2.plot(percentages, bot_outliers, marker="o", label="Bot-Lesion", color="orange")
        ax2.plot(percentages, top_outliers, marker="o", label="Top-Lesion", color="skyblue")
        ax2.set_xlabel("Percentages (%)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Outliers (#)", fontsize=10, fontweight="bold")
        ax2.set_title(f"{model_name}\nOverall Outliers", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid()

        # Labels and title ax1
        ax1.text(percentages[-4], no_lesion - 1.5, "No Lesion Acc.", color="green", fontsize=10, fontweight="bold")
        ax1.text(percentages[-6], chance_threshold + 0.5, "Chance threshold", color="red", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Percentages (%)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
        ax1.set_title(f"{model_name}\nOverall Accuracy", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid()
