
from file_folder_utils import subdirs, subfiles, prepare_nnUNet_file_structure, generate_dataset_json_new
import os
import pandas as pd
from datetime import datetime
import glob
import string



path = r'/data/rj21/nnNet/Data/Task738_MyoSeg_all/Data_all_nii'
out_path = r'/data/rj21/nnNet'
Task = 'Task741_MyoSeg'

datatsets_list = subdirs(path)
patients_list = glob.glob(path+os.sep+'**'+os.sep+'Data.nii.gz', recursive=True)


# prepare the list of all patients and series
scans_list = pd.DataFrame({'Dataset' : []})
scans_list.loc[0, ('Patient')] = ''
scans_list.loc[0, ('Series')] = ''
scans_list.loc[0, ('Data_path')] = ''

i=-1
for _, file in enumerate(patients_list):
    i=i+1
    f = file.split(os.sep)
    scans_list.loc[i, ('Dataset')] = f[7]        
    scans_list.loc[i, ('Patient')] = f[-2]
    scans_list.loc[i, ('Series')] = os.sep.join(f[8:-1])
    scans_list.loc[i, ('Data_path')] = file
    scans_list.sort_values(by=['Patient'])

    # scans_list.loc[i, ('Set')] = 'train'
    # scans_list.loc[i, ('Set')] = 'test'

scans_list.loc[0:600, ('Set')] = 'train'
scans_list.loc[600:len(scans_list), ('Set')] = 'test'

scans_list.to_excel("scans_and_sets_list.xlsx")

training_files_list = scans_list.iloc[0:600]
testing_files_list = scans_list.iloc[600:len(scans_list)]

# training_files_list = training_files_list.iloc[160:170]

final_output_path = os.path.join(out_path, 'nnUNet_raw_data_base', 'nnUNet_raw_data', Task)

prepare_nnUNet_file_structure(training_files_list['Data_path'].values.tolist(),
                              output_path=final_output_path, isTraining=True)
prepare_nnUNet_file_structure(testing_files_list['Data_path'].values.tolist(),
                              output_path=final_output_path, isTraining=False)

generate_dataset_json_new(os.path.join(final_output_path, 'dataset.json'), os.path.join(final_output_path, 'imagesTr'), os.path.join(final_output_path, 'imagesTs'),
                          ['MRI'],  {0: 'background', 1: 'Myocard'}, Task, '?',  'Task741_MyoSeg',
                          "", '0.0')