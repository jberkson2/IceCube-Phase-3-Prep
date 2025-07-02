#Redone Reducer Track

#Reducer Track
import pandas as pd
import numpy as np
import argparse
import json
import os, os.path
from datetime import datetime
from collections import defaultdict

##############################################################################################
#                                       reducer.py
##############################################################################################
# Purpose: Takes in classification list, and reduces into user consensus choices and vote totals
# Usage: python reduce.py <input_dir> <output_dir>
# Author: Andrew Phillips
# Date: 4/2/24

# Code adapted to 2nd iteration due to change in classification data handling
# Madeline Lee
# 10/11/2024

# Code adapted in preparation for 3rd iteration due to needed change for classification
# In particular, different "Tracks" need to be consolidated into single "Track" classification
# User accuracy and time cuts were also implemented here - 6 seconds, and user prompt accuracy
# I also changed the file names from being hardcoded, allowing for more flexibility with naming
# Jonathan Berkson
# 6/22/25
##############################################################################################

class Reducer:
    def __init__(self, input_dir, output_dir, retirement_lim):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim
        self.classif_path = None
        self.subjects_path = None
        self.matched_path = None

if __name__ == '__main__':
    ''' Prompt user for input interactively '''
    lim = int(input("Enter retirement limit (e.g. 20): "))
    accuracy_cut = int(input("Enter minimum user accuracy cutoff (as percent, e.g. 20): "))

    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')

    classif_file = input("Enter classification CSV filename: ").strip()
    subjects_file = input("Enter subjects CSV filename: ").strip()
    matched_file = input("Enter matched data CSV filename: ").strip()
    output_file = input("Enter output filename: ").strip()

    print("\nReducing... (this might take a couple seconds)\n")
    
    # Construct full file paths
    classif_path = os.path.join(input_dir, classif_file)
    subjects_path = os.path.join(input_dir, subjects_file)
    matched_path = os.path.join(input_dir, matched_file)

    # Create reducer instance
    reducer = Reducer(input_dir, output_dir, lim)
    reducer.classif_path = classif_path
    reducer.subjects_path = subjects_path
    reducer.matched_path = matched_path

    def patched_reduce(self):
        lim = self.retirement_lim
        input_dir = self.input_dir
        output_dir = self.output_dir

        # Load all input CSVs
        classif = pd.read_csv(self.classif_path)
        subj = pd.read_csv(self.subjects_path)
        matched = pd.read_csv(self.matched_path)

        # These labels (from ground truth) include specific track/cascade types
        # We will consolidate: throughgoing_track, stopping_track, starting_track → 'TRACK'
        track_labels_truth = {"throughgoing_track", "stopping_track", "starting_track"}
        track_types_user = {"THROUGHGOINGTRACK", "STOPPINGTRACK", "STARTINGTRACK"}

        # Create lookup from subject_id to truth classification label
        truth_lookup = dict(zip(matched['subject_id'], matched['#truth_classification_label']))

        # === USER ACCURACY CALCULATION ===
        # Determine how accurate each user is by comparing their answers to the truth
        user_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for _, row in classif.iterrows():
            user_name = row['user_name']
            subj_id = row['subject_ids']
            annot = json.loads(row['annotations'])
            user_answer = annot[0]['value'][0]['choice']

            # Normalize specific user track answers to 'TRACK'
            if user_answer in track_types_user:
                user_answer = 'TRACK'

            correct_answer = truth_lookup.get(subj_id)
            if correct_answer in track_labels_truth:
                correct_answer = 'TRACK'
            elif correct_answer == "skimming_track":
                correct_answer = 'SKIMMING'
            elif correct_answer in ["contained_hadron_cascade", "unconsained_cascade"]:
                correct_answer = 'CASCADE'

            user_stats[user_name]['total'] += 1
            if correct_answer is not None and user_answer == correct_answer:
                user_stats[user_name]['correct'] += 1

        # Filter users based on input accuracy threshold
        accuracy_threshold = accuracy_cut / 100
        passing_users = {
            user for user, stats in user_stats.items()
            if stats['total'] > 0 and stats['correct'] / stats['total'] >= accuracy_threshold
        }

        # === VOTE COUNTING ===
        # Count the number of votes each subject received in each category (TRACK, CASCADE, SKIMMING)
        subj_ids = np.array(matched['subject_id'])
        subj_dict = {id: {'TRACK': 0, 'CASCADE': 0, 'SKIMMING': 0} for id in subj_ids}

        subj_data = np.array(classif['subject_data'])
        annotations = np.array(classif['annotations'])
        metadata_all = np.array(classif['metadata'])
        user_names = np.array(classif['user_name'])

        for i in range(len(subj_data)):
            user = user_names[i]
            if user not in passing_users:
                continue

            # Parse classification time and skip entries where time ≤ 6 seconds
            try:
                meta = json.loads(metadata_all[i])
                start = datetime.fromisoformat(meta['started_at'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(meta['finished_at'].replace('Z', '+00:00'))
                time_spent = (end - start).total_seconds()
                if time_spent <= 6:
                    continue
            except:
                continue  # Skip bad metadata rows

            metadata = json.loads(subj_data[i])
            annot = json.loads(annotations[i])
            key = int(list(metadata.keys())[0])

            if key in subj_dict:
                user_choice = annot[0]['value'][0]['choice']
                if user_choice in track_types_user:
                    user_choice = 'TRACK'
                if user_choice in ['TRACK', 'CASCADE', 'SKIMMING']:
                    subj_dict[key][user_choice] += 1

        # === FINAL AGGREGATION OF RESULTS ===
        MAX_VOTES = []
        MOST_LIKELY = []
        AGREEMENT = []

        for key in subj_dict:
            max_votes = 0
            most_likely = None
            for cat in subj_dict[key]:
                if subj_dict[key][cat] > max_votes:
                    max_votes = subj_dict[key][cat]
                    most_likely = cat
            MAX_VOTES.append(max_votes)
            MOST_LIKELY.append(most_likely)
            total_votes = sum(subj_dict[key].values())
            AGREEMENT.append(max_votes / total_votes if total_votes > 0 else 0)

        # Prepare and save final output
        data = {
            'subject_id': subj_ids,
            'event_id': subj_ids,  # Duplicate subject_id into event_id for now
            'data.num_votes': MAX_VOTES,
            'data.most_likely': MOST_LIKELY,
            'data.agreement': AGREEMENT
        }

        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(data)
        csv_name = os.path.join(output_dir, f"{output_file}.csv")
        df.to_csv(csv_name, index=False)
        print(f"Reduction complete! Output saved at: \n{csv_name}")
        return csv_name

    from types import MethodType
    reducer.reduce = MethodType(patched_reduce, reducer)
    reducer.reduce()
