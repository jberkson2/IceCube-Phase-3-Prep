import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

##############################################################################################
#                                       reducer.py
##############################################################################################
# Purpose: Reduces classifications into consensus votes (split by all track types separately),
#          applying user accuracy filtering and classification time cutoff.
# Author: Based on Andrew Phillips' logic, updated by Jonathan Berkson
##############################################################################################

class Reducer:
    def __init__(self, input_dir, output_dir, retirement_lim, classif_path, subj_path, matched_path, output_file, accuracy_cut):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim
        self.classif_path = classif_path
        self.subj_path = subj_path
        self.matched_path = matched_path
        self.output_file = output_file
        self.accuracy_cut = accuracy_cut / 100  # Convert percent to fraction

    def reduce(self):
        print("\nReducing... (this might take a couple seconds)")
        classif = pd.read_csv(self.classif_path)
        subj = pd.read_csv(self.subj_path)
        matched = pd.read_csv(self.matched_path)

        subj_ids = np.array(matched['subject_id'])
        subj_dict = {
            id: {
                'THROUGHGOINGTRACK': 0,
                'STOPPINGTRACK': 0,
                'STARTINGTRACK': 0,
                'CASCADE': 0,
                'SKIMMING': 0
            } for id in subj_ids
        }

        # === BUILD TRUTH LABEL LOOKUP FOR ACCURACY ===
        truth_lookup = dict(zip(matched['subject_id'], matched['#truth_classification_label']))
        user_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for _, row in classif.iterrows():
            try:
                user = row['user_name']
                subj_id = row['subject_ids']
                annot = json.loads(row['annotations'])
                user_choice = annot[0]['value'][0]['choice']

                truth = truth_lookup.get(subj_id)
                if truth in ['throughgoing_track', 'throughgoing_bundle']:
                    truth = 'THROUGHGOINGTRACK'
                elif truth in ['stopping_track', 'stopping_bundle']:
                    truth = 'STOPPINGTRACK'
                elif truth in ['starting_track']:
                    truth = 'STARTINGTRACK'
                elif truth in ['contained_em_hadr_cascade', 'contained_hadron_cascade']:
                    truth = 'CASCADE'
                elif truth in ['skimming_track', 'uncontained_cascade']:
                    truth = 'SKIMMING'

                if user_choice == 'TRACK':
                    answers = annot[0]['value'][0].get('answers', {})
                    user_choice = answers.get('WHATTYPEOFTRACKISIT', None)

                user_stats[user]['total'] += 1
                if user_choice == truth:
                    user_stats[user]['correct'] += 1
            except:
                continue

        passing_users = {
            user for user, stat in user_stats.items()
            if stat['total'] > 0 and stat['correct'] / stat['total'] >= self.accuracy_cut
        }

        # === ACTUAL VOTE COUNTING ===
        subj_data = np.array(classif['subject_data'])
        annotations = np.array(classif['annotations'])
        metadata_all = np.array(classif['metadata'])
        user_names = np.array(classif['user_name'])

        for i in range(len(subj_data)):
            try:
                user = user_names[i]
                if user not in passing_users:
                    continue

                meta = json.loads(metadata_all[i])
                start = datetime.fromisoformat(meta['started_at'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(meta['finished_at'].replace('Z', '+00:00'))
                time_spent = (end - start).total_seconds()
                if time_spent <= 6:
                    continue

                metadata = json.loads(subj_data[i])
                annot = json.loads(annotations[i])
                key = int(list(metadata.keys())[0])
                if key not in subj_dict:
                    continue

                user_choice = annot[0]['value'][0]['choice']
                if user_choice in ['CASCADE', 'SKIMMING']:
                    subj_dict[key][user_choice] += 1
                elif user_choice == 'TRACK':
                    track_type = annot[0]['value'][0].get('answers', {}).get('WHATTYPEOFTRACKISIT')
                    if track_type in subj_dict[key]:
                        subj_dict[key][track_type] += 1
            except:
                continue

        # === FINAL AGGREGATION ===
        MAX_VOTES = []
        MOST_LIKELY = []
        AGREEMENT = []

        for key in subj_ids:
            counts = subj_dict[key]
            max_votes = 0
            most_likely = None
            total_votes = sum(counts.values())

            for label, val in counts.items():
                if val > max_votes:
                    max_votes = val
                    most_likely = label

            MAX_VOTES.append(max_votes)
            MOST_LIKELY.append(most_likely)
            AGREEMENT.append(max_votes / total_votes if total_votes > 0 else 0)

        data = {
            'subject_id': subj_ids,
            'event_id': subj_ids,
            'data.num_votes': MAX_VOTES,
            'data.most_likely': MOST_LIKELY,
            'data.agreement': AGREEMENT
        }

        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, f"{self.output_file}.csv")
        df.to_csv(output_path, index=False)
        print(f"\nReduction complete! Output saved at:\n{output_path}")
        return output_path


if __name__ == '__main__':
    lim = int(input("Enter retirement limit (e.g. 20): "))
    accuracy_cut = int(input("Enter minimum user accuracy cutoff (percent, e.g. 20): "))

    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')

    classif_file = input("Enter classification CSV filename: ").strip()
    subj_file = input("Enter subjects CSV filename: ").strip()
    matched_file = input("Enter matched data CSV filename: ").strip()
    output_file = input("Enter desired output filename (no .csv): ").strip()

    reducer = Reducer(
        input_dir=input_dir,
        output_dir=output_dir,
        retirement_lim=lim,
        classif_path=os.path.join(input_dir, classif_file),
        subj_path=os.path.join(input_dir, subj_file),
        matched_path=os.path.join(input_dir, matched_file),
        output_file=output_file,
        accuracy_cut=accuracy_cut
    )
    reducer.reduce()