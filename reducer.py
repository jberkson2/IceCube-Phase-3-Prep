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
        self.output_file = None

if __name__ == '__main__':
    # User input prompts
    lim = int(input("Enter retirement limit (e.g. 20): "))
    accuracy_cut = int(input("Enter minimum user accuracy cutoff (as percent, e.g. 20): "))

    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')

    classif_file = input("Enter classification CSV filename: ").strip()
    subjects_file = input("Enter subjects CSV filename: ").strip()
    matched_file = input("Enter matched data CSV filename: ").strip()
    output_file = input("Enter output filename (without .csv): ").strip()

    print("\nReducing... (this might take a couple seconds)\n")

    # Full file paths
    classif_path = os.path.join(input_dir, classif_file)
    subjects_path = os.path.join(input_dir, subjects_file)
    matched_path = os.path.join(input_dir, matched_file)

    # Create reducer instance and assign paths
    reducer = Reducer(input_dir, output_dir, lim)
    reducer.classif_path = classif_path
    reducer.subjects_path = subjects_path
    reducer.matched_path = matched_path
    reducer.output_file = output_file

    def patched_reduce(self):
        lim = self.retirement_lim
        accuracy_threshold = accuracy_cut / 100
        output_dir = self.output_dir

        # Load CSVs
        classif = pd.read_csv(self.classif_path)
        subj = pd.read_csv(self.subjects_path)
        matched = pd.read_csv(self.matched_path)

        # Truth labels for user accuracy checking - condensed to 'TRACK'
        track_truth_labels = {"throughgoing_track","starting_track","stopping_track","throughgoing_bundle","stopping_bundle"}
        skimming_truth_labels = {"skimming_track","uncontained_cascade"}
        cascade_truth_labels = {"contained_em_hadr_cascade","contained_hadron_cascade"}

        # Create subject_id → truth classification lookup (condensed)
        truth_lookup = {}
        for sid, truth_label in zip(matched['subject_id'], matched['#truth_classification_label']):
            if truth_label in track_truth_labels:
                truth_lookup[sid] = 'TRACK'
            elif truth_label in skimming_truth_labels:
                truth_lookup[sid] = 'SKIMMING'
            elif truth_label in cascade_truth_labels:
                truth_lookup[sid] = 'CASCADE'
            else:
                truth_lookup[sid] = None  # Unknown or unclassified

        # === USER ACCURACY CALCULATION ===
        user_stats = defaultdict(lambda: {'correct':0, 'total':0})

        for _, row in classif.iterrows():
            user_name = row['user_name']
            subj_id = row['subject_ids']
            annot = json.loads(row['annotations'])
            user_choice = annot[0]['value'][0]['choice']

            # Normalize all track types to 'TRACK' for user accuracy
            if user_choice in ['THROUGHGOINGTRACK', 'STARTINGTRACK', 'STOPPINGTRACK', 'TRACK']:
                user_choice = 'TRACK'

            correct_answer = truth_lookup.get(subj_id, None)

            user_stats[user_name]['total'] += 1
            if correct_answer is not None and user_choice == correct_answer:
                user_stats[user_name]['correct'] += 1

        # Filter users by accuracy cutoff
        passing_users = {
            user for user, stats in user_stats.items()
            if stats['total'] > 0 and (stats['correct'] / stats['total']) >= accuracy_threshold
        }

        # === VOTE COUNTING ===
        subj_ids = np.array(matched['subject_id'])
        subj_dict = {id: {'TRACK':0, 'CASCADE':0, 'SKIMMING':0} for id in subj_ids}

        subj_data = np.array(classif['subject_data'])
        annotations = np.array(classif['annotations'])
        metadata_all = np.array(classif['metadata'])
        user_names = np.array(classif['user_name'])

        count_votes = 0
        count_skipped_time = 0
        count_skipped_user = 0
        count_skipped_key = 0
        count_skipped_unknown_choice = 0

        for i in range(len(subj_data)):
            user = user_names[i]
            if user not in passing_users:
                count_skipped_user += 1
                continue

            try:
                meta = json.loads(metadata_all[i])
                start = datetime.fromisoformat(meta['started_at'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(meta['finished_at'].replace('Z', '+00:00'))
                time_spent = (end - start).total_seconds()
                if time_spent <= 6:
                    count_skipped_time += 1
                    continue
            except Exception:
                count_skipped_time += 1
                continue

            try:
                metadata = json.loads(subj_data[i])
                key = int(list(metadata.keys())[0])
            except Exception:
                count_skipped_key += 1
                continue

            if key not in subj_dict:
                count_skipped_key += 1
                continue

            try:
                annot = json.loads(annotations[i])
                user_choice = annot[0]['value'][0]['choice']
            except Exception:
                count_skipped_unknown_choice += 1
                continue

            # Combine all track subtypes into 'TRACK'
            if user_choice in ['THROUGHGOINGTRACK', 'STARTINGTRACK', 'STOPPINGTRACK', 'TRACK']:
                user_choice = 'TRACK'

            if user_choice in ['TRACK', 'CASCADE', 'SKIMMING']:
                subj_dict[key][user_choice] += 1
                count_votes += 1
            else:
                count_skipped_unknown_choice += 1

        # === FINAL AGGREGATION OF RESULTS ===
        MAX_VOTES = []
        MOST_LIKELY = []
        AGREEMENT = []

        for key in subj_dict:
            votes = subj_dict[key]
            max_votes = 0
            most_likely = None
            for cat, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    most_likely = cat
            MAX_VOTES.append(max_votes)
            MOST_LIKELY.append(most_likely)
            total_votes = sum(votes.values())
            AGREEMENT.append(max_votes / total_votes if total_votes > 0 else 0)

        # Save output CSV
        data = {
            'subject_id': subj_ids,
            'event_id': subj_ids,  # duplicate subject_id into event_id for now
            'data.num_votes': MAX_VOTES,
            'data.most_likely': MOST_LIKELY,
            'data.agreement': AGREEMENT
        }

        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(data)
        csv_name = os.path.join(output_dir, f"{self.output_file}.csv")
        df.to_csv(csv_name, index=False)

        print(f"Reduction complete! Output saved at:\n{csv_name}")
        print(f"Votes counted: {count_votes}")
        print(f"Skipped due to time ≤ 6s or bad metadata: {count_skipped_time}")

        return csv_name

    from types import MethodType
    reducer.reduce = MethodType(patched_reduce, reducer)
    reducer.reduce()
