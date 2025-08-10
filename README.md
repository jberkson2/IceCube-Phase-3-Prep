# IceCube-Phase-3-Prep
Summer 2025 work prepping for IceCube Phase 3. The work done here is a redone reducer and consolidator, along with a new counter vote checker. Changes are noted in each section. The code here is based upon [Maddie Lee's](https://github.com/leemadeline75/IceCube-Phase-2-Data-Analysis/tree/main)  work for Phase 2 analysis.

## FILES NEEDED TO RUN PROGRAMS
#### Classification CSV
Classification csv includes classification level data: classification_id, user_name, user_id, user_ip, workflow_id, workflow_name, workflow_version, created_at, metadata, annotations, subject_data, and subject_ids. This file is used in reducer.py to do time cuts, user accuracy cuts, and reduction.
#### Subjects CSV
Subjects csv includes subject level data: subject_id, project_id, workflow_id, metadata, etc. This file is used in reducer.py as a list of all subject ids.
#### Matched Data CSV
Matched data csv includes usefu data, as well as DNN data: subject_id, truth_classification, pred_skim, pred_cascade, pred_tgtrack, pred_starttrack, pred_stoptrack, energy, zenith, oneweight, signal_charge, bg_charge, qtot, qratio, log10_max_charge, #truth_classification_label, max_score_val, idx_max_score, ntn_category, etc. This file is used in both reducer.py and consolidator.py.
#### Reduced Data CSV
This is the output file from reducer.py. This contains data for use in the consolidator: subject_id, event_id, data.num_votes, data.most_likely, data.agreement.

#### For clarification on different files and columns, check out [Maddie Lee's README](https://github.com/leemadeline75/IceCube-Phase-2-Data-Analysis/tree/main#)

## UPDATED .PY PROGRAMS

Please note the .py files with 5option in the front do the same thing as the .py files detailed below, but without combining starting, stopping, and throughgoing tracks into a singular "track" category. Not combining these tracks may effect calculated user accuracy and event agreement, so be aware of that when comparing output files and plots.

## reducer.py

The needed input files for the reducer are the Classification, Subjects, and Matched Data files. While this code is similar to the previous reducer code, there were a couple of things that needed to be changed or added. My main goal was to consolidate the three different track variations into a single general track. Along with this, user accuracy and time cuts were implemented here. With each event video being 6 seconds in Zooniverse, only user classifications made after watching the full videos were considered. Metadata from the input files allowed the start and end times to be calculated, allowing for a time_spent threshold to be calculated and applied. 

Similarly, user accuracy was also calculated within this code by, for each user, calculating the amount of correct answers by comparing user choice to established "truth" event classifications. A users' amount of correct answers was then divided by the total amount of events classified by the user. For ease of use, the user accuracy cut is prompted when running the reducer.

The user accuracy prompt brings up a change that I wanted to implement. The old reducer had file names and directories hardcoded. This code has updated that to be user entered, allowing for greater flexibility.

## consolidator.py

The needed input files for the consolidator are the Reduced Data and Matched Data files. The previous consolidator was within the do_analysis.py file, so I separated it out to be its own individual .py file. The consolidator combines the user choices and the DNN information into one file.

The major thing changed in this version of the consolidator was combining the different track types into an individual general track. Comments are made within the consolidator with the process, but the general structure was to combine the DNN prediction columns for the different tracks into a pred_track column, and changing other individual columns to reflect this consolidation. The consolidator also has the agreement cut that is specified by user input. This only keeps events for which the users "agree" on what the event should be classified as above the user-indicated threshold.

Again, this version of the consolidator has user prompts for file names and directories rather than them being hardcoded.

## Plotter.py

The needed input file for the plotter is the consolidated file - which is the output from consolidator.py. The plotter creates two confusion matrices - DNN vs Truth and User vs Truth. Users indicate the input and output directories in addition to the names of the plots.

Example matrices:

<img width="1000" height="800" alt="60Ac90Ag-DNN" src="https://github.com/user-attachments/assets/c4018796-6be9-4452-b72a-4fa70bfb29da" />

<img width="1000" height="800" alt="60Ac90Ag-User" src="https://github.com/user-attachments/assets/09c5e576-851b-49ee-a0b7-e84c9fd763dc" />
