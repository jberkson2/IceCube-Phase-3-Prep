import os
import pandas as pd
from types import MethodType

##############################################################################################
#                                       consolidator.py
##############################################################################################
# Purpose: Consolidates user classifications and simulation predictions into a unified file
# Usage: python consolidator.py (interactive input prompts)
# Author: Jonathan Berkson (updated 1/14/26)
##############################################################################################

class Consolidator:
    def __init__(self, input_dir, output_dir, retirement_lim, agreement_cut):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim  # Minimum votes required per subject
        self.agreement_cut = agreement_cut    # Minimum agreement fraction required
        self.classif_path = None
        self.matched_path = None
        self.output_file = None


if __name__ == '__main__':
    # === Prompt user for input ===
    lim = int(input("Enter retirement limit (minimum votes per subject, e.g. 20): "))
    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')
    classif_file = input("Enter reduced CSV filename (output from Reducer.py): ").strip()
    matched_file = input("Enter matched_sim_data CSV filename: ").strip()
    output_file = input("Enter output filename (without .csv): ").strip()
    agreement_cut = float(input("Enter agreement cutoff (e.g. 0.6 to keep rows with >=60% agreement): "))

    print("\nConsolidating... (this might take a few seconds)\n")

    # === Construct full file paths ===
    classif_path = os.path.join(input_dir, classif_file)
    matched_path = os.path.join(input_dir, matched_file)

    # === Create Consolidator instance ===
    consolidator = Consolidator(input_dir, output_dir, lim, agreement_cut)
    consolidator.classif_path = classif_path
    consolidator.matched_path = matched_path
    consolidator.output_file = output_file

    # === Define consolidate method dynamically ===
    def patched_consolidate(self):
        # Load CSVs
        user_data = pd.read_csv(self.classif_path)
        user_data.columns = user_data.columns.str.strip()
        dnn_sim_data = pd.read_csv(self.matched_path)

        # Check required columns
        required_cols = ['subject_id', 'event_id', 'data.num_votes', 'data.most_likely', 'data.agreement']
        missing = [col for col in required_cols if col not in user_data.columns]
        if missing:
            raise KeyError(f"Missing expected columns in classification file: {missing}")

        # === Prepare user DataFrame ===
        subj_user_data = pd.DataFrame({
            'subject_id': user_data['subject_id'],
            'filename': [f"subject_{sid}_event_{eid}.txt" for sid, eid in zip(user_data['subject_id'], user_data['event_id'])],
            'run': [None] * len(user_data),
            'event': user_data['event_id'],
            'data.num_votes': user_data['data.num_votes'],
            'data.most_likely': user_data['data.most_likely'],
            'data.agreement': user_data['data.agreement']
        })

        # === Apply retirement limit: keep only subjects with enough votes ===
        if self.retirement_lim is not None and self.retirement_lim > 0:
            subj_user_data = subj_user_data[subj_user_data['data.num_votes'] >= self.retirement_lim]

        # === Merge with DNN simulation data ===
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

        # === Unify track labels ===
        track_types = {"THROUGHGOINGTRACK", "STARTINGTRACK", "STOPPINGTRACK"}
        cdf['data.most_likely'] = cdf['data.most_likely'].apply(
            lambda x: 'TRACK' if isinstance(x, str) and x.upper() in track_types else x
        )

        # === Consolidate DNN track predictions ===
        cdf['pred_track'] = cdf[['pred_tgtrack', 'pred_starttrack', 'pred_stoptrack']].sum(axis=1)
        cdf.drop(columns=['pred_tgtrack', 'pred_starttrack', 'pred_stoptrack'], inplace=True)
        insert_at = cdf.columns.get_loc('pred_cascade') + 1
        cdf.insert(insert_at, 'pred_track', cdf.pop('pred_track'))

        # === Update max_score_val ===
        cdf['max_score_val'] = cdf[['pred_skim', 'pred_cascade', 'pred_track']].max(axis=1)

        # === Update idx_max_score robustly ===
        score_columns = ['pred_skim', 'pred_cascade', 'pred_track']
        label_mapping = {
            'pred_skim': 'SKIMMING',
            'pred_cascade': 'CASCADE',
            'pred_track': 'TRACK'
        }

        def safe_label(row):
            vals = {col: row[col] for col in score_columns if pd.notnull(row[col])}
            if not vals:
                return None
            return label_mapping[max(vals, key=vals.get)]

        cdf['idx_max_score'] = cdf.apply(safe_label, axis=1)

        # === Map ntn_category and collapse track types ===
        ntn_label_mapping = {
            0: 'SKIMMING',
            1: 'CASCADE',
            2: 'THROUGHGOINGTRACK',
            3: 'STARTINGTRACK',
            4: 'STOPPINGTRACK'
        }
        cdf['ntn_category'] = cdf['ntn_category'].map(ntn_label_mapping)
        cdf['ntn_category'] = cdf['ntn_category'].apply(
            lambda x: 'TRACK' if isinstance(x, str) and x in track_types else x
        )

        # === Compute accuracies ===
        cdf['user_accuracy'] = cdf.apply(
            lambda row: int(row['data.most_likely'] == row['ntn_category']),
            axis=1
        )
        cdf['DNN_accuracy'] = cdf.apply(
            lambda row: int(row['idx_max_score'] == row['ntn_category']),
            axis=1
        )

        # === Save output ===
        os.makedirs(self.output_dir, exist_ok=True)
        csv_name = os.path.join(self.output_dir, f"{self.output_file}.csv")
        cdf.to_csv(csv_name, index=False)

        return csv_name

    # === Bind and execute the patched method ===
    consolidator.consolidate = MethodType(patched_consolidate, consolidator)
    csv_path = consolidator.consolidate()

    print(f"Consolidation complete. Output saved at:\n{csv_path}")
