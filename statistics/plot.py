import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast
import json
from .stats import compute_diff, compute_mean_and_std

def count_and_plot_multiple(data, columns, ax, title):
    """
    Count occurrences of categorical values across multiple columns and plot them on the given axis.

    Parameters:
    - data: pandas DataFrame containing the data.
    - columns: List of column names to analyze.
    - ax: The Matplotlib axis on which to plot.
    - title: Title for the subplot.
    """
    # Prepare a DataFrame to aggregate counts for each column
    count_data = pd.DataFrame()

    for col in columns:
        count_data[col] = data[col].value_counts()

    # Define the positions and width for bars
    categories = count_data.index
    x = np.arange(len(categories))  # the label locations
    bar_width = 0.35 / len(columns)  # Adjust bar width based on the number of columns
    spacing = bar_width / 5  # Small spacing between bars in a group

    # Create the bar plot on the specified axis
    for i, col in enumerate(columns):
        ax.bar(x + i * (bar_width + spacing), count_data[col], width=bar_width, label=col)

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel("Unique Values")
    ax.set_ylabel("Count")
    ax.set_xticks(x + (len(columns) - 1) * (bar_width + spacing) / 2)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend(title="Variables")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Example usage: Creating 3 plots and saving them in one file
def save_combined_plot(data, columns, masks, titles, filename):
    """
    Generates 3 subplots for different masks and saves them in a single file.

    Parameters:
    - data: pandas DataFrame containing the data.
    - columns: List of column names to analyze.
    - masks: List of masks to filter the data.
    - titles: List of titles for the subplots.
    - filename: Name of the file to save the plots (e.g., 'plots.png').
    """
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows, 1 column

    for i, (mask, title) in enumerate(zip(masks, titles)):
        count_and_plot_multiple(data[mask], columns, axes[i], title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

from datetime import datetime
import os
import pandas as pd
import ast

def freq_plot(df, model_name, csv_file, subfolder):
    """
    Processes a DataFrame to compute occurrences of each answer and generates frequency plots
    for specific question types. Saves the plots in the 'freq' folder with the current date.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        model_name (str): The name of the model for titling the plots.
        csv_file (str): The name of the current CSV file being processed.
        subfolder (str): The path to the folder containing the CSV file.

    Returns:
        None
    """
    # Create a copy of the DataFrame and parse the "plot_info" column
    new_tab = df.copy()
    new_tab["plot_info"] = new_tab["plot_info"].apply(ast.literal_eval)

    # Add the "original_location" column from "plot_info"
    new_tab["original_location"] = new_tab["plot_info"].apply(lambda x: x["original_place"])

    # Dynamically identify all 'predict' columns
    predict_columns = [col for col in df.columns if col.startswith("predict")]
    columns = predict_columns + ["answer"]

    # Define masks for different question types
    mask_loc = (~new_tab["is_closed_question"]) & (~new_tab["is_tertiary"])
    mask_yes = new_tab["is_closed_question"]
    mask_ord = new_tab["is_tertiary"]

    # Apply transformations to the 'predict' columns based on the masks
    for col in columns:
        # Transform for location-based questions
        new_tab.loc[mask_loc, col] = new_tab.loc[mask_loc].apply(
            lambda row: "original_loc" if row[col] == row["original_location"] else
                        ("new_location" if row[col] == row["new_location"] else None), axis=1
        )

        # Transform for ordinal questions
        new_tab.loc[mask_ord, col] = new_tab.loc[mask_ord].apply(
            lambda row: row[col].split(" ")[0], axis=1
        )

    # Define masks and titles for plotting
    masks = [mask_loc, mask_yes, mask_ord]
    titles = [
        f"{model_name}: Location Questions (OpenToM)",
        f"{model_name}: Closed Questions (OpenToM)",
        f"{model_name}: Tertiary Questions (OpenToM)"
    ]

    # # Create the 'freq' folder if it doesn't exist
    # freq_folder = os.path.join(subfolder, "freq")
    # os.makedirs(freq_folder, exist_ok=True)

    # Get the current date for the filename
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Define the output path for the plots
    plot_filename = f"freq_{os.path.splitext(csv_file)[0]}.png"
    plot_path = os.path.join(subfolder, plot_filename)

    # Generate and save combined plots
    save_combined_plot(new_tab, columns, masks, titles, plot_path)

    # Indicate that the plot was saved
    print(f"Frequency plots saved to: {plot_path}")


def barplot_diff(input_data, input_type="dataframe", filter_variable=None, output_file=None, title=None):
    """
    Plots a bar plot comparing differences across categories in the specified filter variable,
    with optional standard deviation error bars, and saves the plot to a file.

    Args:
        input_data: Either a pandas DataFrame or a JSON file path containing the data.
        input_type (str): Type of input, either "dataframe" or "json".
        filter_variable (str): The column name used for filtering categories in compute_diff.
        output_file (str): Path to save the plot instead of displaying it.

    Returns:
        None
    """
    if filter_variable is None:
        raise ValueError("The filter_variable must be specified.")

    # Determine the source of data
    if input_type == "dataframe":
        # Compute differences from the DataFrame
        diff = compute_diff(input_data, filter_variable)
    elif input_type == "json":
        # Load differences from the JSON file
        with open(input_data, "r") as json_file:
            diff = json.load(json_file)
    else:
        raise ValueError("Invalid input_type. Must be 'dataframe' or 'json'.")

    # Extract categories dynamically
    print(diff)
    categories = list(diff.keys())
    means_topk = []
    stds_topk = []
    means_random = []
    stds_random = []

    has_random = False  # Check if random values exist
    for category in categories:
        # Compute mean and std for "top_k"
        mean_topk, std_topk = compute_mean_and_std(diff[category]["top_k"])
        means_topk.append(mean_topk * 100)
        stds_topk.append(std_topk * 100)

        # Compute mean and std for "random" if present
        if "random" in diff[category]:
            has_random = True
            mean_random, std_random = compute_mean_and_std(diff[category]["random"])
            means_random.append(mean_random * 100)
            stds_random.append(std_random * 100)

    # Set up positions for the bars
    x = np.arange(len(categories))
    width = 0.35

    # Plotting
    fig, ax = plt.subplots()

    # Plot top-k bars with error bars
    ax.bar(x - width / 2 if has_random else x, means_topk, width, yerr=stds_topk, label="top-k", color="blue", capsize=5)

    # Plot random bars with error bars if random values are present
    if has_random:
        ax.bar(x + width / 2, means_random, width, yerr=stds_random, label="Random", color="grey", capsize=5)

    # Add labels, title, and legend
    ax.set_xlabel(filter_variable)
    ax.set_ylabel('Diff Performance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if title is not None:
        ax.set_title(title)

    # Add legend dynamically based on what is plotted
    ax.legend() if has_random else ax.legend(["top-k"])

    plt.grid(True)
    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory

# def plot_all_csv_in_result(input_path, output_save):
#     """
#     Traverse the 'result' directory, process CSV files in each folder, and save the plots
#     in the respective folders.

#     Args:
#         input_path (str): Path to the "result" directory.

#     Returns:
#         None
#     """
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"The path {input_path} does not exist.")

#     # List all subfolders
#     subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]

#     for subfolder in subfolders:
#         # Look for CSV files in the subfolder
#         csv_files = [file for file in os.listdir(subfolder) if file.endswith(".csv")]
#         for csv_file in csv_files:
#             csv_path = os.path.join(subfolder, csv_file)

#             # Load the CSV file into a DataFrame
#             df = pd.read_csv(csv_path)
            
#             # Determine the filter variable based on the filename
#             if csv_file.startswith(("OpenToM", "Variant_OpenToM")):
#                 df.drop([4884, 4904], inplace=True)
#                 df["cands"] =  df["cands"].apply(ast.literal_eval)
#                 df["is_tertiary"] = df['answer'].str.startswith(('less', 'equally', 'more'))
#                 df["Qtype"] = df.apply(lambda row: "[Yes, No]" if row["is_closed_question"]
#                                     else "[Less, Equally, More]" if row["is_tertiary"]
#                                     else "[Initial loc., New loc.]", axis=1)
#                 # Determine the benchmark name based on the file name
#                 benchmark_name = "Variant_OpenToM" if csv_file.startswith("Variant_OpenToM") else "OpenToM"
#                 filter_variable = "Qtype"
#             elif csv_file.startswith("ToMi"):
#                 filter_variable = "falseTrueBelief"
#                 benchmark_name = "ToMi"
#             else:
#                 print(f"Skipping file {csv_file} as it doesn't match criteria.")
#                 continue

#             # Define output file path for the plot
#             plot_path = os.path.join(subfolder, f"{os.path.splitext(csv_file)[0]}_plot.png")

#             print(f"Processing file: {csv_path} with filter variable: {filter_variable}")
#             print(f"Saving plot to: {plot_path}")

#             # Generate and save the plot
#             model_name = subfolder.split("/")[-1]
#             title = f"{model_name}: {benchmark_name}"
#             try:
#                 barplot_diff(df, input_type="dataframe", filter_variable=filter_variable, output_file=plot_path, title=title)
#             except Exception as e:
#                 print(f"Error processing file {csv_path}: {e}")
            
#             if benchmark_name == "OpenToM":
#                 process_and_plot_answers(df, model_name, csv_file, subfolder)


import os
import pandas as pd
import ast
from datetime import datetime

def plot_all_csv_in_result(input_path):
    """
    Traverse the 'result' directory, process CSV files in the 'data' folder of each subfolder, 
    and save the plots in respective 'plot/<benchmark_name>' folders with the current date.

    Args:
        input_path (str): Path to the "result" directory.

    Returns:
        None
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The path {input_path} does not exist.")

    # List all subfolders in the input path
    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]

    # Current date for naming the plots
    current_date = datetime.now().strftime("%d-%m-%Y")

    for subfolder in subfolders:
        data_folder = os.path.join(subfolder, "data")
        plot_folder = os.path.join(subfolder, "plot")

        # Ensure the 'plot' folder exists
        os.makedirs(plot_folder, exist_ok=True)

        # Look for CSV files in the 'data' folder
        if not os.path.exists(data_folder):
            print(f"Data folder {data_folder} does not exist. Skipping...")
            continue

        csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]
        for csv_file in csv_files:
            csv_path = os.path.join(data_folder, csv_file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Determine the filter variable and benchmark name based on the filename
            if csv_file.startswith(("OpenToM", "Variant_OpenToM")):
                df.drop([4884, 4904], inplace=True)
                df["cands"] = df["cands"].apply(ast.literal_eval)
                df["is_tertiary"] = df['answer'].str.startswith(('less', 'equally', 'more'))
                df["Qtype"] = df.apply(lambda row: "[Yes, No]" if row["is_closed_question"]
                                       else "[Less, Equally, More]" if row["is_tertiary"]
                                       else "[Initial loc., New loc.]", axis=1)
                benchmark_name = "Variant_OpenToM" if csv_file.startswith("Variant_OpenToM") else "OpenToM"
                filter_variable = "Qtype"
            elif csv_file.startswith("ToMi"):
                filter_variable = "falseTrueBelief"
                benchmark_name = "ToMi"
            else:
                print(f"Skipping file {csv_file} as it doesn't match criteria.")
                continue

            # Create a subfolder for the benchmark in the 'plot' folder
            benchmark_plot_folder = os.path.join(plot_folder, benchmark_name)
            os.makedirs(benchmark_plot_folder, exist_ok=True)

            # Define the output file path for the plot
            plot_filename = f"{os.path.splitext(csv_file)[0]}_{current_date}.png"
            plot_path = os.path.join(benchmark_plot_folder, plot_filename)

            print(f"Processing file: {csv_path} with filter variable: {filter_variable}")
            print(f"Saving plot to: {plot_path}")

            # Generate and save the plot
            model_name = subfolder.split("/")[-1]
            title = f"{model_name}: {benchmark_name}"
            try:
                barplot_diff(df, input_type="dataframe", filter_variable=filter_variable, output_file=plot_path, title=title)
            except Exception as e:
                print(f"Error processing file {csv_path} for plot: {e}")

            # Generate and save the frequency plot
            if benchmark_name in ["OpenToM", "Variant_OpenToM"]:
                try:
                    freq_plot(df, model_name, csv_file, benchmark_plot_folder)
                except Exception as e:
                    print(f"Error in freq_plot for file {csv_path}: {e}")