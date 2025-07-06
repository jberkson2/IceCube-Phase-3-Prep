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
    def __init__(self, input_dir, output_dir, retirement_lim):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim
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

    print("\nConsolidating... (this might take a few seconds)\n")
    
    # Construct full file paths
    classif_path = os.path.join(input_dir, classif_file)
    matched_path = os.path.join(input_dir, matched_file)

    # Create Consolidator instance
    consolidator = Consolidator(input_dir, output_dir, lim)
    consolidator.classif_path = classif_path
    consolidator.matched_path = matched_path
    consolidator.output_file = output_file

    # Define the consolidate method to attach dynamically
    def patched_consolidate(self):
        user_data = pd.read_csv(self.classif_path)
        user_data.columns = user_data.columns.str.strip()  # <-- Fix potential whitespace issue in column names

        dnn_sim_data = pd.read_csv(self.matched_path)

        # Check that all required columns exist
        required_cols = ['subject_id', 'event_id', 'data.num_votes', 'data.most_likely', 'data.agreement']
        missing = [col for col in required_cols if col not in user_data.columns]
        if missing:
            raise KeyError(f"Missing expected columns in classification file: {missing}")

        # Prepare user data
        subj_user_data = pd.DataFrame({
            'subject_id': user_data['subject_id'],
            'filename': [f"subject_{sid}_event_{eid}.txt" for sid, eid in zip(user_data['subject_id'], user_data['event_id'])],
            'run': [None] * len(user_data),
            'event': user_data['event_id'],
            'data.num_votes': user_data['data.num_votes'],
            'data.most_likely': user_data['data.most_likely'],
            'data.agreement': user_data['data.agreement']
        })

        # Prepare DNN/simulation data
        dnn_data = dnn_sim_data[[
            'subject_id', 'filename', 'run', 'event', 'truth_classification',
            'pred_skim', 'pred_cascade', 'pred_tgtrack', 'pred_starttrack', 'pred_stoptrack',
            'energy', 'zenith', 'oneweight', 'signal_charge', 'bg_charge',
            'qratio', 'qtot', 'max_score_val', 'idx_max_score', 'ntn_category'
        ]].copy()

        # Merge both DataFrames
        cdf = pd.merge(subj_user_data, dnn_data, on='subject_id', how='outer')

        # Drop duplicate columns
        cdf.drop(columns=['filename_x', 'run_x', 'event_x', 'run_y', 'event_y'], inplace=True, errors='ignore')

        # === Start custom logic ===

        # Defines what the track types are that I will want to change
        track_types = {"THROUGHGOINGTRACK", "STARTINGTRACK", "STOPPINGTRACK"}

        # First thing I want to do is change different tracks into general 'TRACK'
        cdf['data.most_likely'] = cdf['data.most_likely'].apply(lambda x: 'TRACK' if x in track_types else x)

        # Second thing I want to do is consolidate pred_tgtrack, pred_starttrack, pred_stoptrack into a new column
        # I will call this new column pred_track, and it will be the sum of the three individual tracks
        # I will then place this new column where the first of the three original columns were
        cdf['pred_track'] = cdf[['pred_tgtrack', 'pred_starttrack', 'pred_stoptrack']].sum(axis=1)
        cdf.drop(columns=['pred_tgtrack', 'pred_starttrack', 'pred_stoptrack'], inplace=True)
        insert_at = cdf.columns.get_loc('pred_cascade') + 1
        cdf.insert(insert_at, 'pred_track', cdf.pop('pred_track'))

        # Third thing I want to do is adjust max_score_val
        # Changes value of max_score_val to be the max of the per-row values of the different pred_ categories
        cdf['max_score_val'] = cdf[['pred_skim', 'pred_cascade', 'pred_track']].max(axis=1)

        # Fourth thing I want to do is similar to the first change: change idx_max_score to just 'TRACK'
        # Fourth thing I want to do is update idx_max_score to reflect the category with the highest predicted score
        score_columns = ['pred_skim', 'pred_cascade', 'pred_track']
        label_mapping = {
            'pred_skim': 'SKIMMING',
            'pred_cascade': 'CASCADE',
            'pred_track': 'TRACK'
        }

        # Find the column with the highest value per row, then map to the corresponding category
        cdf['idx_max_score'] = cdf[score_columns].idxmax(axis=1).map(label_mapping)

        # Map numeric values in ntn_category to their corresponding category names
        ntn_label_mapping = {
            0: 'SKIMMING',
            1: 'CASCADE',
            2: 'THROUGHGOINGTRACK',
            3: 'STARTINGTRACK',
            4: 'STOPPINGTRACK'
        }

        cdf['ntn_category'] = cdf['ntn_category'].map(ntn_label_mapping).fillna(cdf['ntn_category'])

        # Fifth thing I want to do is similar again: change ntn_category to just 'TRACK'
        cdf['ntn_category'] = cdf['ntn_category'].apply(lambda x: 'TRACK' if x in track_types else x)

        # Sixth thing: compute classification accuracy for user and DNN
        # Set to 1 if predicted category matches true label, else 0

        # User accuracy: does user's most likely classification match truth?
        cdf['user_accuracy'] = cdf.apply(
            lambda row: int(row['data.most_likely'] == row['ntn_category']),
            axis=1
        )

        # DNN accuracy: does DNN max-score classification match truth?
        cdf['DNN_accuracy'] = cdf.apply(
            lambda row: int(row['idx_max_score'] == row['ntn_category']),
            axis=1
        )

        # === End custom logic ===

        # Save the final DataFrame
        os.makedirs(self.output_dir, exist_ok=True)
        csv_name = os.path.join(self.output_dir, f"{self.output_file}.csv")
        cdf.to_csv(csv_name, index=False)
        return csv_name

    # Bind and execute
    consolidator.consolidate = MethodType(patched_consolidate, consolidator)
    csv_path = consolidator.consolidate()

    print(f" Consolidation complete. Output saved at: \n{csv_path}")
