import numpy as np
import os
import pyedflib
import pandas as pd
import re
import gc

from func_compute_seizure_nonseizure_hour_time_record import *
from get_common_channels_over_all_files import *


time_length_segment = 6  
majority = 4
allowed_max_nonseizure_timelenth_in_seizure_segment = time_length_segment - majority
stride = 3

seizure_time_hour_record_dict, non_seizure_time_hour_record_dict = func_compute_seizure_nonseizure_hour_time_records(allowed_max_nonseizure_timelenth_in_seizure_segment)

num_sei_hours = 0
num_non_sei_hours = 0

dir = '/home/xya238/CHB_MIT_EEGdata/'

folder_list = ['chb01/','chb02/','chb03/','chb04/','chb05/','chb06/','chb07/','chb08/',
               'chb09/','chb10/','chb11/','chb12/','chb13/','chb14/','chb15/','chb16/',
               'chb17/','chb18/','chb19/','chb20/','chb21/','chb22/','chb23/','chb24/']

store_path = '/scratch/xya238/CHB_MIT_process_results/hour_data_samples_overlap_6s_seg_4s_major_fullsei_start_end_pre_post_nonsei/'


#_______________________________________________________________________________________________________________________________________
#to achieve number of seizure time hour records which equals to 3600 seconds, and number of non-seizure time hour records
#_______________________________________________________________________________________________________________________________________

for folder_x in folder_list :
    for file_x in list(seizure_time_hour_record_dict[folder_x].keys()) :
        sei_time_hour_list_x = seizure_time_hour_record_dict[folder_x][file_x]

        for sei_time_hour_x in sei_time_hour_list_x :
            if sei_time_hour_x[1] - sei_time_hour_x[0] + 1 == 3600 :
                num_sei_hours += 1
            else:
                print(file_x, sei_time_hour_x)

print('num_sei_hours: ', num_sei_hours)


for folder_y in folder_list :
    for file_y in list(non_seizure_time_hour_record_dict[folder_y].keys()) :
        non_sei_time_hour_list_y = non_seizure_time_hour_record_dict[folder_y][file_y]

        for non_sei_time_hour_y in non_sei_time_hour_list_y :
            if non_sei_time_hour_y[1] - non_sei_time_hour_y[0] + 1 == 3600 :
                num_non_sei_hours += 1
            else:
                print(file_y, non_sei_time_hour_y)

print('num_non_sei_hours: ', num_non_sei_hours)


#_______________________________________________________________________________________________________________________________________
#to achieve common labels list
#_______________________________________________________________________________________________________________________________________

common_labels_list = get_labels_excluding_three_files()

for folder_x in folder_list :  
    dir_X = dir + folder_x
    files = os.listdir(dir_X)
    file_list_for_labels = list(filter(lambda x:x.endswith('.edf'),files))
    file_list_for_labels.sort()

    for file_x in file_list_for_labels :
        f_1 = pyedflib.EdfReader(dir_X + file_x)

        signal_headers_1 = f_1.getSignalHeaders()
        label_list_a_file = []
        for ch in range(len(signal_headers_1)):
            label_list_a_file.append(signal_headers_1[ch]['label'])
        
        for label in common_labels_list :
            count = label_list_a_file.count(label)
            if count >= 2 :
                common_labels_list.remove(label)
                
        f_1._close()


common_labels_list.sort()

print('After changing, common_labels_list:',common_labels_list)


#_______________________________________________________________________________________________________________________________________
#to achieve seizure start time and end time of each seizure
#_______________________________________________________________________________________________________________________________________

seizure_time_dict_folder_key = {}

for folder_x in folder_list :
    dir_X_1 = dir + folder_x

    seizure_time_dict_file_key = {}

    if folder_x != 'chb24/':
        with open(dir_X_1 + dir_X_1[-6:-1] + '-summary.txt','r') as f:    
            line = f.readline()

            while line:
        
                line = line.strip()
                if line[0:10]=='File Name:':
                    file_name = line[11:]
                    f.readline()  #read file start time
                    f.readline()  # read file end time
                    line = f.readline().strip() # read seizure number
            
                    assert line[0:27]=='Number of Seizures in File:'

                    if int(line[28:])==0:
                        seizure_time_dict_file_key[file_name] = None
                    else:
                        assert int(line[28:]) > 0
                        seizure_time_list = []
                        for i in range(int(line[28:])):
                            line_1 = f.readline().strip()
                            seizure_start_time = int(re.findall(r"\d+\.?\d*",line_1)[-1])
                            line_2 = f.readline().strip()
                            seizure_end_time = int(re.findall(r"\d+\.?\d*",line_2)[-1])
                            seizure_time_list.append((seizure_start_time,seizure_end_time))
                        seizure_time_dict_file_key[file_name] = seizure_time_list
                line = f.readline()
    else:
        seizure_time_dict_file_key = {
            'chb24_01.edf':[(480,505),(2451,2476)],
            'chb24_02.edf':None,
            'chb24_03.edf':[(231,260),(2883,2908)],
            'chb24_04.edf':[(1088,1120),(1411,1438),(1745,1764)],
            'chb24_05.edf':None,
            'chb24_06.edf':[(1229,1253)],
            'chb24_07.edf':[(38,60)],
            'chb24_08.edf':None,
            'chb24_09.edf':[(1745,1764)],
            'chb24_10.edf':None,
            'chb24_11.edf':[(3527,3597)],
            'chb24_12.edf':None,
            'chb24_13.edf':[(3288,3304)],
            'chb24_14.edf':[(1939,1966)],
            'chb24_15.edf':[(3552,3569)],
            'chb24_16.edf':None,
            'chb24_17.edf':[(3515,3581)],
            'chb24_18.edf':None,
            'chb24_19.edf':None,
            'chb24_20.edf':None,
            'chb24_21.edf':[(2804,2872)],
            'chb24_22.edf':None
    
            }
    seizure_time_dict_folder_key[folder_x] = seizure_time_dict_file_key


#_______________________________________________________________________________________________________________
#to achieve hour data samples with seizure according to folder
#_______________________________________________________________________________________________________________

for folder_x in folder_list :
    seizure_hour_data_sample_folder = []
    #non_seizure_hour_data_sample_folder = []

    file_list_with_seizure_hour = list(seizure_time_hour_record_dict[folder_x].keys())
    file_list_with_seizure_hour.sort()

    for file_x in file_list_with_seizure_hour :
        sei_time_hour_list_x = seizure_time_hour_record_dict[folder_x][file_x]

        f_x = pyedflib.EdfReader(dir + folder_x + file_x)
        signal_header_x = f_x.getSignalHeaders()

        if len(sei_time_hour_list_x) > 0 :
            sei_time_list_a_file_x = seizure_time_dict_folder_key[folder_x][file_x]
            assert len(sei_time_list_a_file_x) >= 1

        for sei_time_hour_x in sei_time_hour_list_x :
            if sei_time_hour_x[1] - sei_time_hour_x[0] + 1 == 3600 :
                sei_time_hour_start_x = sei_time_hour_x[0]
                sei_time_hour_end_x = sei_time_hour_x[1]

                sei_one_hour_data_x = []
                num_segment = 0
                
                for i in range(sei_time_hour_start_x, sei_time_hour_end_x - (time_length_segment - 1) + 1, stride) :
                    window_x = list(range(i, i + time_length_segment))
                    set_window_x = set(window_x)
                    assert len(window_x) == time_length_segment
                                                                                
                    #_______________________________________________________________________________________________________________
                                        
                    num_non_empty_intersection_happen = 0     

                    start_seizure_length_list = []
                    full_seizure_length_list = []
                    end_seizure_length_list = []

                    for seizure_duration_x in sei_time_list_a_file_x :                    
                        seizure_time_x_list = list(range(seizure_duration_x[0], seizure_duration_x[1]))
                        set_x = set(seizure_time_x_list)

                        if set_window_x & set_x != set() :
                            num_non_empty_intersection_happen += 1
                            num_common_time = len(set_window_x & set_x)
                    
                            if set_window_x.issubset(set_x) :
                                full_seizure_length_list.append(num_common_time)
                            elif not window_x[0] in set_x :
                                start_seizure_length_list.append(num_common_time)                            
                            elif not window_x[-1] in set_x :
                                end_seizure_length_list.append(num_common_time)

                    assert num_non_empty_intersection_happen <= 1
                    assert len(full_seizure_length_list) + len(start_seizure_length_list) + len(end_seizure_length_list) == num_non_empty_intersection_happen

                    if full_seizure_length_list != [] :
                        seg_label_x = 'full_seizure'

                    if start_seizure_length_list != [] :
                        seg_label_x = 'seizure_start_' + str(start_seizure_length_list[0]) + 's'

                    if end_seizure_length_list != [] :
                        seg_label_x = 'seizure_end_' + str(end_seizure_length_list[0]) + 's'

                    if num_non_empty_intersection_happen == 0 :
                        num_seizure_x = len(sei_time_list_a_file_x)
                        j=0
                        num_blonging_cases = 0
                                                                        
                        while j <= num_seizure_x - 1 :
                            if j == 0 :
                                if window_x[-1] < sei_time_list_a_file_x[j][0] :
                                    distance = sei_time_list_a_file_x[j][0] - window_x[-1]
                                    seg_label_x = 'pre_ictal_' + str(distance) + 's'
                                    num_blonging_cases += 1                                    
                                    
                            if j == num_seizure_x - 1 :
                                if window_x[0] > sei_time_list_a_file_x[j][-1] - 1 :
                                    distance = window_x[0] - (sei_time_list_a_file_x[j][-1] - 1)
                                    seg_label_x = 'post_ictal_' + str(distance) + 's'
                                    num_blonging_cases += 1
                                    
                                if num_seizure_x >= 2 :
                                    if window_x[0] > sei_time_list_a_file_x[j-1][-1] - 1 and window_x[-1] < sei_time_list_a_file_x[j][0] :
                                        distance_1 = window_x[0] - (sei_time_list_a_file_x[j-1][-1] - 1)
                                        distance_2 = sei_time_list_a_file_x[j][0] - window_x[-1]
                                        if distance_1 < distance_2 :
                                            seg_label_x = 'post_ictal_' + str(distance_1) + 's'
                                            num_blonging_cases += 1
                                        elif distance_2 < distance_1 :
                                            seg_label_x = 'pre_ictal_' + str(distance_2) + 's'
                                            num_blonging_cases += 1
                                        else :
                                            seg_label_x = 'pre_ictal_' + str(distance_2) + 's'
                                            num_blonging_cases += 1
                                    
                            if j != 0 and j != num_seizure_x - 1 :
                                if window_x[0] > sei_time_list_a_file_x[j-1][-1] - 1 and window_x[-1] < sei_time_list_a_file_x[j][0] :
                                    distance_1 = window_x[0] - (sei_time_list_a_file_x[j-1][-1] - 1)
                                    distance_2 = sei_time_list_a_file_x[j][0] - window_x[-1]
                                    if distance_1 < distance_2 :
                                        seg_label_x = 'post_ictal_' + str(distance_1) + 's'
                                        num_blonging_cases += 1
                                        
                                        
                                    elif distance_2 < distance_1 :
                                        seg_label_x = 'pre_ictal_' + str(distance_2) + 's'
                                        num_blonging_cases += 1
                                        
                                        
                                    else :
                                        seg_label_x = 'pre_ictal_' + str(distance_2) + 's'
                                        num_blonging_cases += 1
                                        
                            j += 1
                            
                                                
                        assert num_blonging_cases == 1

                    
                    #___________________________________________________________________________________________
                        
                    segment_data_dict_chan_key = {}
                    time_step_list_x = []
                    for time_x in window_x :
                        time_step_list_x = time_step_list_x + [k + time_x * 256 for k in range(0, 256)]
                                            
                    
                    for ch in range(len(signal_header_x)) :
                        ch_label_x = signal_header_x[ch]['label']
                       
                        if ch_label_x in common_labels_list :
                            chan_data = f_x.readSignal(chn=ch,start=0,n=None)
                            assert len(chan_data.shape) == 1
                            assert f_x.getSampleFrequency(ch) == 256
        
                            segment_a_chan = []
                            for time_step_x in time_step_list_x :
                                segment_a_chan.append(chan_data[time_step_x])
                            segment_a_chan.append(seg_label_x)
                            segment_a_chan.append(file_x)

                            segment_a_chan = np.array(segment_a_chan)
                            segment_a_chan = np.expand_dims(segment_a_chan, axis=1)

                            segment_data_dict_chan_key[ch_label_x] = segment_a_chan

                    segment_array = segment_data_dict_chan_key[common_labels_list[0]]
                    for j in range(1, len(common_labels_list)) :
                        segment_array = np.concatenate([segment_array, segment_data_dict_chan_key[common_labels_list[j]]], axis=1)

                    num_segment += 1
                    #print('num_segment: ',num_segment,'segment_array shape: ', segment_array.shape)
                    sei_one_hour_data_x.append(segment_array)

                assert sei_one_hour_data_x != []
                
                print('len(sei_one_hour_data_x): ',len(sei_one_hour_data_x))
                
                seizure_hour_data_sample_folder.append(sei_one_hour_data_x)
                
                print('len(seizure_hour_data_sample_folder): ', len(seizure_hour_data_sample_folder))

                del sei_one_hour_data_x
                del segment_array
                gc.collect()

    seizure_hour_data_sample_folder = np.array(seizure_hour_data_sample_folder)
    print(folder_x, 'seizure_hour_data_sample_folder array shape: ', seizure_hour_data_sample_folder.shape)

    np.save(store_path + 'hour_data_samples_overlap_' + str(time_length_segment) + 's_' +
            'seg_' + str(majority) + 's_major_' + str(stride) + 's_stride_' +
            'fullsei_start_end_pre_post_' + folder_x[0:-1] + '_with_seizure.npy', seizure_hour_data_sample_folder)

    del seizure_hour_data_sample_folder
    gc.collect()


'''
#____________________________________________________________________________________________________________________
#to achieve hour data without seizure according to folder, selecting one hour data from each edf file without seizure
#____________________________________________________________________________________________________________________
    
for folder_y in folder_list :
    non_seizure_hour_data_sample_folder = []

    file_list_with_non_sei_hour = list(non_seizure_time_hour_record_dict[folder_y].keys())
    file_list_with_non_sei_hour.sort()

    for file_y in file_list_with_non_sei_hour :
        if seizure_time_dict_folder_key[folder_y][file_y] is not None :
            continue
        else :
            non_sei_time_hour_list_y = non_seizure_time_hour_record_dict[folder_y][file_y]

            f_y = pyedflib.EdfReader(dir + folder_y + file_y)
            signal_header_y = f_y.getSignalHeaders()
            
            for non_sei_time_hour_y in non_sei_time_hour_list_y :
                non_sei_time_hour_start = non_sei_time_hour_y[0]
                non_sei_time_hour_end = non_sei_time_hour_y[1]
                if non_sei_time_hour_end - non_sei_time_hour_start + 1 != 3600 :
                    continue
                else :
                    non_sei_hour_data_y = []
                    for i in range(non_sei_time_hour_start, non_sei_time_hour_end - (time_length_segment - 1) + 1, stride) :
                        window_y = list(range(i, i + time_length_segment))

                        assert len(window_y) == time_length_segment

                        segment_data_dict_chan_key_y = {}

                        time_step_list_y = []
                        for time_y in window_y :
                            time_step_list_y = time_step_list_y + [k + time_y * 256 for k in range(0,256)]

                        for ch_y in range(len(signal_header_y)) :
                            ch_label_y = signal_header_y[ch_y]['label']
                       
                            if ch_label_y in common_labels_list :
                                chan_data_y = f_y.readSignal(chn=ch_y,start=0,n=None)
                                assert len(chan_data_y.shape) == 1
                                assert f_y.getSampleFrequency(ch_y) == 256

                                segment_a_chan_y = [chan_data_y[time_step_y] for time_step_y in time_step_list_y]
                                segment_a_chan_y.append('non_seizure')
                                segment_a_chan_y.append(file_y)
                                segment_a_chan_y = np.array(segment_a_chan_y)
                                segment_a_chan_y = np.expand_dims(segment_a_chan_y, axis=1)
                                
                                segment_data_dict_chan_key_y[ch_label_y] = segment_a_chan_y

                                del segment_a_chan_y
                                gc.collect()
                        
                        segment_array_y = segment_data_dict_chan_key_y[common_labels_list[0]]
                        for j in range(1, len(common_labels_list)) :
                            segment_array_y = np.concatenate([segment_array_y,segment_data_dict_chan_key_y[common_labels_list[j]]], axis=1)

                        del segment_data_dict_chan_key_y
                        gc.collect()

                        non_sei_hour_data_y.append(segment_array_y)

                        del segment_array_y
                        gc.collect()

                    non_seizure_hour_data_sample_folder.append(non_sei_hour_data_y)

                    del non_sei_hour_data_y
                    gc.collect()

                    break

    non_seizure_hour_data_sample_folder = np.array(non_seizure_hour_data_sample_folder)
    print(folder_y, 'non_seizure_hour_data_sample_folder array shape: ', non_seizure_hour_data_sample_folder.shape)
    
    np.save(store_path + 'hour_data_samples_overlap_' + str(time_length_segment) + 's_' +
            'seg_' + str(majority) + 's_major_' + str(stride) + 's_stride_' +
            'non_sei_' + folder_y[0:-1] + '_without_seizure.npy', non_seizure_hour_data_sample_folder)

    del non_seizure_hour_data_sample_folder
    gc.collect()
                        
           
'''    
    
                    
                        
                                    
                            
                        

                
                
                























