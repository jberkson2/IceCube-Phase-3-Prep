import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

##############################################################################################
#                                       Plotter.py
##############################################################################################
# Purpose: Generates confusion matrices between user, DNN, and truth classification labels
# Usage: python Plotter.py (interactive input prompts)
# Author: Jonathan Berkson
# Date: 7/8/25
##############################################################################################

def plot_confusion_matrices(df, outdir, dnn_filename, user_filename):
    # Internal categories (used for data)
    categories = ['SKIMMING', 'CASCADE', 'TRACK']
    # Display labels (used for plotting)
    nice_labels = ['Skimming', 'Cascade', 'Track']

    # Ensure columns are categorical with correct ordering
    for col in ['data.most_likely', 'idx_max_score', 'ntn_category']:
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True)

    # Create output directory
    os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)

    def make_conf_matrix(x_label, y_label, x_data, y_data, filename):
        # Align category order
        x_data = pd.Categorical(x_data, categories=categories, ordered=True)
        y_data = pd.Categorical(y_data, categories=categories, ordered=True)

        # Counts and column-normalized percentages
        counts = pd.crosstab(y_data, x_data, rownames=['True'], colnames=['Predicted'])
        norm = counts.div(counts.sum(axis=0), axis=1) * 100  # Normalize by column
        norm = norm.round(1)

        # Column totals for annotation
        col_totals = counts.sum(axis=0)
        annot = norm.astype(str) + '%' + '\n' + counts.astype(str) + '/' + col_totals.astype(str)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.rcParams['font.family'] = 'Helvetica'

        sns.heatmap(norm, annot=annot, fmt='', annot_kws={"size": 15},
                    cmap='Blues', xticklabels=nice_labels, yticklabels=nice_labels,
                    vmin=0.0, vmax=100.0, cbar_kws={'label': 'Percentage'}, ax=ax)

        plt.ylabel(y_label, fontsize=20, labelpad=10)
        plt.xlabel(x_label, fontsize=20, labelpad=10)
        ax.tick_params(axis='both', labelsize=14)

        plt.title(f"{y_label} vs {x_label}", fontsize=22, pad=20)
        plt.savefig(os.path.join(outdir, 'plots', filename), bbox_inches='tight')
        plt.close()

    # === Confusion Matrix Plots ===
    make_conf_matrix('MC Truth Label', 'DNN Max Category',
                     df['idx_max_score'], df['ntn_category'], dnn_filename)

    make_conf_matrix('MC Truth Label', 'User Agreement',
                     df['data.most_likely'], df['ntn_category'], user_filename)

    # Print raw (unnormalized) confusion matrix
    raw_counts = pd.crosstab(df['data.most_likely'], df['ntn_category'])
    print("\nRaw Confusion Matrix (User vs MC Truth):")
    print(raw_counts)


if __name__ == '__main__':
    print("==== Confusion Matrix Generator ====")

    input_dir = input("Enter input directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff): ").strip('"')
    output_dir = input("Enter output directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff\\output_data): ").strip('"')
    classif_file = input("Enter consolidated CSV filename (output from Consolidator.py): ").strip()

    dnn_plot_filename = input("Enter filename for DNN vs MC Truth plot (e.g. dnn_mc_confusion.png): ").strip()
    user_plot_filename = input("Enter filename for User vs MC Truth plot (e.g. user_mc_confusion.png): ").strip()

    print("\nGenerating confusion matrices... (this might take a few seconds)\n")

    file_path = os.path.join(input_dir, classif_file)
    df = pd.read_csv(file_path)

    required_columns = {'data.most_likely', 'idx_max_score', 'ntn_category'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    plot_confusion_matrices(df, output_dir, dnn_plot_filename, user_plot_filename)

    print("\nDone! Plots saved in:", os.path.join(output_dir, 'plots'))
