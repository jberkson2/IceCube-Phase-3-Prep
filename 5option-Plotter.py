import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

expected_categories = [
    "SKIMMING",
    "CASCADE",
    "THROUGHGOINGTRACK",
    "STARTINGTRACK",
    "STOPPINGTRACK"
]

def compute_fraction_matrix_user(df):
    df = df.dropna(subset=["ntn_category", "data.most_likely"])
    df = df[df["ntn_category"].isin(expected_categories) & df["data.most_likely"].isin(expected_categories)]

    count_matrix = pd.crosstab(df["data.most_likely"], df["ntn_category"])
    count_matrix = count_matrix.reindex(index=expected_categories, columns=expected_categories, fill_value=0)
    column_totals = count_matrix.sum(axis=0)

    fraction_matrix = pd.DataFrame(index=expected_categories, columns=expected_categories)
    annotation_matrix = pd.DataFrame(index=expected_categories, columns=expected_categories)

    for row_cat in expected_categories:
        for col_cat in expected_categories:
            numerator = count_matrix.loc[row_cat, col_cat]
            denominator = column_totals[col_cat]
            denom_safe = denominator if denominator > 0 else 1  # avoid division by zero
            frac_str = f"{numerator}/{denominator}"
            pct_str = f"{numerator / denom_safe * 100:.1f}%"
            fraction_matrix.loc[row_cat, col_cat] = frac_str
            annotation_matrix.loc[row_cat, col_cat] = f"{pct_str}\n{frac_str}"  # percentage on top

    return fraction_matrix, annotation_matrix

def compute_fraction_matrix_dnn(df):
    df = df.dropna(subset=["ntn_category", "idx_max_score"])
    df = df[df["ntn_category"].isin(expected_categories) & df["idx_max_score"].isin(expected_categories)]

    count_matrix = pd.crosstab(df["idx_max_score"], df["ntn_category"])
    count_matrix = count_matrix.reindex(index=expected_categories, columns=expected_categories, fill_value=0)
    column_totals = count_matrix.sum(axis=0)

    fraction_matrix = pd.DataFrame(index=expected_categories, columns=expected_categories)
    annotation_matrix = pd.DataFrame(index=expected_categories, columns=expected_categories)

    for row_cat in expected_categories:
        for col_cat in expected_categories:
            numerator = count_matrix.loc[row_cat, col_cat]
            denominator = column_totals[col_cat]
            denom_safe = denominator if denominator > 0 else 1
            frac_str = f"{numerator}/{denominator}"
            pct_str = f"{numerator / denom_safe * 100:.1f}%"
            fraction_matrix.loc[row_cat, col_cat] = frac_str
            annotation_matrix.loc[row_cat, col_cat] = f"{pct_str}\n{frac_str}"  # percentage on top

    return fraction_matrix, annotation_matrix

def convert_to_numeric(fraction_matrix):
    numeric_matrix = fraction_matrix.copy()
    for row in numeric_matrix.index:
        for col in numeric_matrix.columns:
            num, denom = fraction_matrix.loc[row, col].split('/')
            denom = int(denom) if int(denom) > 0 else 1
            numeric_matrix.loc[row, col] = int(num) / denom
    return numeric_matrix.astype(float)

def plot_confusion_matrix(numeric_matrix, annotation_matrix, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(
        numeric_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap="Blues",
        cbar_kws={'label': 'Fraction of Truth Category'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0, vmax=1  # Fix color scale from 0 to 1 (0% to 100%)
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("==== 5-Category Confusion Matrix Generator ====")

    input_dir = input("Enter input directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff): ").strip('"')
    output_dir = input("Enter output directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff\\output_data): ").strip('"')
    classif_file = input("Enter consolidated CSV filename (e.g. consolidated-July8-individualtracks.csv): ").strip()

    user_plot_filename = input("Enter filename for User vs MC Truth plot (e.g. user_mc_confusion_5cat.png): ").strip()
    dnn_plot_filename = input("Enter filename for DNN vs MC Truth plot (e.g. dnn_mc_confusion_5cat.png): ").strip()

    input_path = os.path.join(input_dir, classif_file)
    output_path_user = os.path.join(output_dir, user_plot_filename)
    output_path_dnn = os.path.join(output_dir, dnn_plot_filename)

    # Load data
    df = pd.read_csv(input_path)

    # User confusion matrix
    frac_user, annot_user = compute_fraction_matrix_user(df)
    print("\nUser Classification Fraction Matrix:\n")
    print(frac_user)
    num_user = convert_to_numeric(frac_user)
    plot_confusion_matrix(
        num_user, annot_user,
        title="User Classification vs. Truth (5 Categories)",
        xlabel="Ground Truth: ntn_category",
        ylabel="User Prediction: data.most_likely",
        output_path=output_path_user
    )

    # DNN confusion matrix
    frac_dnn, annot_dnn = compute_fraction_matrix_dnn(df)
    print("\nDNN Classification Fraction Matrix:\n")
    print(frac_dnn)
    num_dnn = convert_to_numeric(frac_dnn)
    plot_confusion_matrix(
        num_dnn, annot_dnn,
        title="DNN Classification vs. Truth (5 Categories)",
        xlabel="Ground Truth: ntn_category",
        ylabel="DNN Prediction: idx_max_score",
        output_path=output_path_dnn
    )
