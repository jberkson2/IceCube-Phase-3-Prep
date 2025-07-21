import pandas as pd
import os
from types import MethodType

##############################################################################################
#                                       consolidator.py
##############################################################################################
# Purpose: Consolidates user classifications and simulation predictions into a unified file
# Usage: python consolidator.py (interactive input prompts)
# Author: Jonathan Berkson
# Date: 6/26/25
##############################################################################################

class Consolidator:
    def __init__(self, input_dir, output_dir, retirement_lim, agreement_cut):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim
        self.agreement_cut = agreement_cut
        self.classif_path = None
        self.matched_path = None
        self.output_file = None


if __name__ == '__main__':
    ''' Prompt user for input interactively '''
    lim = int(input("Enter retirement limit (e.g. 20): "))

    input_dir = input("Enter input directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff): ").strip('"')
    output_dir = input("Enter output directory path (e.g. C:\\Users\\jonat\\Documents\\IceCube Research Stuff\\output_data): ").strip('"')

    classif_file = input("Enter reduced CSV filename (output from Reducer.py): ").strip()
    matched_file = input("Enter matched_sim_data CSV filename: ").strip()
    output_file = input("Enter what you would like the output file to be called: ").strip()

    agreement_cut = float(input("Enter agreement cutoff (e.g. 0.6 to keep rows with >=60% agreement): "))

    print("\nConsolidating... (this might take a few seconds)\n")

    # Construct full file paths
    classif_path = os.path.join(input_dir, classif_file)
    matched_path = os.path.join(input_dir, matched_file)

    # Create Consolidator instance
    consolidator = Consolidator(input_dir, output_dir, lim, agreement_cut)
    consolidator.classif_path = classif_path
    consolidator.matched_path = matched_path
    consolidator.output_file = output_file

    # Define the consolidate method to attach dynamically
    def patched_consolidate(self):
        user_data = pd.read_csv(self.classif_path)
        user_data.columns = user_data.columns.str.strip()

        dnn_sim_data = pd.read_csv(self.matched_path)

        required_cols = ['subject_id', 'event_id', 'data.num_votes', 'data.most_likely', 'data.agreement']
        missing = [col for col in required_cols if col not in user_data.columns]
        if missing:
            raise KeyError(f"Missing expected columns in classification file: {missing}")

        subj_user_data = pd.DataFrame({
            'subject_id': user_data['subject_id'],
            'filename': [f"subject_{sid}_event_{eid}.txt" for sid, eid in zip(user_data['subject_id'], user_data['event_id'])],
            'run': [None] * len(user_data),
            'event': user_data['event_id'],
            'data.num_votes': user_data['data.num_votes'],
            'data.most_likely': user_data['data.most_likely'],
            'data.agreement': user_data['data.agreement']
        })

        dnn_data = dnn_sim_data[[
            'subject_id', 'filename', 'run', 'event', 'truth_classification',
            'pred_skim', 'pred_cascade', 'pred_tgtrack', 'pred_starttrack', 'pred_stoptrack',
            'energy', 'zenith', 'oneweight', 'signal_charge', 'bg_charge',
            'qratio', 'qtot', 'max_score_val', 'idx_max_score', 'ntn_category'
        ]].copy()

        cdf = pd.merge(subj_user_data, dnn_data, on='subject_id', how='outer')
        cdf.drop(columns=['filename_x', 'run_x', 'event_x', 'run_y', 'event_y'], inplace=True, errors='ignore')

        # === Apply agreement_cut after merge ===
        if 'data.agreement' in cdf.columns:
            cdf = cdf[cdf['data.agreement'] >= self.agreement_cut]

        # === Begin logic cleanup ===

        # 4. Update idx_max_score robustly
        score_columns = ['pred_skim', 'pred_cascade', 'pred_tgtrack','pred_starttrack','pred_stoptrack']
        label_mapping = {
            'pred_skim': 'SKIMMING',
            'pred_cascade': 'CASCADE',
            'pred_tgtrack': 'THROUGHGOINGTRACK',
            'pred_starttrack': 'STARTINGTRACK',
            'pred_stoptrack': 'STOPPINGTRACK'
        }

        def safe_label(row):
            vals = {col: row[col] for col in score_columns if pd.notnull(row[col])}
            if not vals:
                return None
            return label_mapping[max(vals, key=vals.get)]

        cdf['idx_max_score'] = cdf.apply(safe_label, axis=1)

        # 5. Map ntn_category to strings
        ntn_label_mapping = {
            0: 'SKIMMING',
            1: 'CASCADE',
            2: 'THROUGHGOINGTRACK',
            3: 'STARTINGTRACK',
            4: 'STOPPINGTRACK'
        }

        cdf['ntn_category'] = cdf['ntn_category'].map(ntn_label_mapping)

        # 6. Accuracy calculations
        cdf['user_accuracy'] = cdf.apply(
            lambda row: int(row['data.most_likely'] == row['ntn_category']),
            axis=1
        )
        cdf['DNN_accuracy'] = cdf.apply(
            lambda row: int(row['idx_max_score'] == row['ntn_category']),
            axis=1
        )

        # === End logic ===

        # Save output
        os.makedirs(self.output_dir, exist_ok=True)
        csv_name = os.path.join(self.output_dir, f"{self.output_file}.csv")
        cdf.to_csv(csv_name, index=False)
        return csv_name

    # Bind and execute
    consolidator.consolidate = MethodType(patched_consolidate, consolidator)
    csv_path = consolidator.consolidate()

    print(f" Consolidation complete. Output saved at: \n{csv_path}")
