
from file_folder_utils import subdirs, subfiles, prepare_nnUNet_file_structure, generate_dataset_json_new
import os
import pandas as pd
from datetime import datetime
import glob



path = r'/data/rj21/nnNet/Data/Task737_Clin_MyoSeg/Data_all_nii'
out_path = r'/data/rj21/nnNet'
Task = 'Task737_Clin_MyoSeg'

# datatsets_list = subdirs(path)
# patients_list = glob.glob(path+os.sep+'**'+os.sep+'Data.nii.gz', recursive=True)

patients_list = subdirs(path)

# prepare the list of all patients and series
scans_list = pd.DataFrame({'Patient' : []})
scans_list.loc[0, ('Series')] = ''
# scans_list.loc[0, ('ScanDate')] = datetime.strptime('19000101', '%Y%m%d')
scans_list.loc[0, ('Data_path')] = ''


i = 0;
for patient in patients_list:
    patients_series_list = subdirs(patient)
    for patient_serie in patients_series_list:
        patients_series_files_list = subfiles(patient_serie)
        # for file in patients_series_files_list:
        file = patients_series_files_list[0]
        scans_list.loc[i, ('Patient')] = os.path.basename(patient)
        scans_list.loc[i, ('Series')] = os.path.basename(patient_serie).split('_')[1]
        scans_list.loc[i, ('Data_path')] = file
        i = i+1

# sort the data by Patient name and ScanDate + select testing dataset as ceil(total_number_of_pateints*(1/5))
# cca 20 % of patients will be in testing set (cca 10 % with lesion and cca 10 % without lesion) - no matter how many series are available

scans_list.sort_values(by=['Patient'])
# scans_list = scans_list.drop(scans_list[scans_list.Series == '5325QUCO_20100108'].index) # removed patient (see mail from Renda 21/03/2022 16:26)
scans_list.loc[0:67, ('Set')] = 'train'
scans_list.loc[67:len(scans_list), ('Set')] = 'test'
scans_list.to_excel("scans_and_sets_list.xlsx")

training_files_list = scans_list.iloc[0:62]
testing_files_list = scans_list.iloc[62:len(scans_list)]

final_output_path = os.path.join(out_path, 'nnUNet_raw_data_base', 'nnUNet_raw_data', Task)

prepare_nnUNet_file_structure(training_files_list['Data_path'].values.tolist(),
                              output_path=final_output_path, isTraining=True)
prepare_nnUNet_file_structure(testing_files_list['Data_path'].values.tolist(),
                              output_path=final_output_path, isTraining=False)

generate_dataset_json_new(os.path.join(final_output_path, 'dataset.json'), os.path.join(final_output_path, 'imagesTr'), os.path.join(final_output_path, 'imagesTs'),
                          ['MRI'],  {0: 'background', 1: 'Myocard'}, Task, '?',  'Task737_Clin_MyoSeg_test_1',
                          "", '0.0')