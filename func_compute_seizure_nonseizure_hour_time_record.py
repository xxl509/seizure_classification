import numpy as np
import os
import pyedflib
import pandas as pd
import re
import gc


def compute_duration_3600_7200(duration_x, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x):
    assert duration_x >= 3600 and duration_x < 7200
    assert duration_x - 1 + base_x >= seizure_list_end_x - 1
    assert base_x <= seizure_list_start_x
    
    seizure_hour_list_x = []
    non_seizure_hour_list_x = []
    if duration_x - 1 + base_x <= seizure_list_end_x - 1 + timestep_x :
        seizure_hour_list_x.append((duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x))
    elif seizure_list_start_x - timestep_x <= base_x :
        seizure_hour_list_x.append((base_x, base_x + 3600 - 1))
    elif duration_x - 1 + base_x - (seizure_list_end_x - 1 + timestep_x) >= 3600 - (seizure_list_end_x - 1 + timestep_x - (seizure_list_start_x - timestep_x) + 1) :
        seizure_hour_list_x.append((seizure_list_start_x - timestep_x, seizure_list_start_x - timestep_x + 3600 - 1))
    elif seizure_list_start_x - timestep_x - base_x >= 3600 - (seizure_list_end_x - 1 + timestep_x - (seizure_list_start_x - timestep_x) + 1) :
        seizure_hour_list_x.append((seizure_list_end_x - 1 + timestep_x - 3600 + 1, seizure_list_end_x - 1 + timestep_x))
    else:
        seizure_hour_list_x.append((base_x, base_x + 3600 - 1))

    return seizure_hour_list_x, non_seizure_hour_list_x


def compute_duration_7200_10800(duration_x, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x):
    assert duration_x >= 7200 and duration_x < 10800
    assert duration_x - 1 + base_x >= seizure_list_end_x - 1
    assert base_x <= seizure_list_start_x
    
    seizure_hour_list_x = []
    non_seizure_hour_list_x = []
    if duration_x - 1 + base_x <= seizure_list_end_x - 1 + timestep_x :
        seizure_hour_list_x = seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        
    elif seizure_list_start_x - timestep_x <= base_x :
        seizure_hour_list_x = seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]
        
    elif duration_x - 1 + base_x - (seizure_list_end_x -1 + timestep_x) >= 3600 :
        seizure_hour_3600_7200, non_seizure_hour_3600_7200 = compute_duration_3600_7200(duration_x - 3600, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_3600_7200
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_3600_7200
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]

    elif seizure_list_start_x - timestep_x - 1 - base_x + 1 >= 3600 :
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        seizure_hour_3600_7200, non_seizure_hour_3600_7200 = compute_duration_3600_7200(duration_x - 3600, base_x + 3600, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_3600_7200
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_3600_7200

    elif duration_x - 1 + base_x - (seizure_list_end_x - 1 + timestep_x) >= 3600 - (seizure_list_end_x - 1 + timestep_x - (seizure_list_start_x - timestep_x) + 1) :
        seizure_hour_list_x = seizure_hour_list_x + [(seizure_list_start_x - timestep_x, seizure_list_start_x - timestep_x + 3600 - 1)]

    elif seizure_list_start_x - timestep_x - 1 - base_x + 1 >= 3600 - (seizure_list_end_x - 1 + timestep_x - (seizure_list_start_x - timestep_x) + 1) :
        seizure_hour_list_x = seizure_hour_list_x + [(seizure_list_end_x - 1 + timestep_x - 3600 + 1, seizure_list_end_x - 1 + timestep_x)]

    return seizure_hour_list_x, non_seizure_hour_list_x


def compute_duration_10800_14400(duration_x, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x):
    assert duration_x >= 10800 and duration_x < 14400
    assert duration_x - 1 + base_x >= seizure_list_end_x - 1
    assert base_x <= seizure_list_start_x
    
    seizure_hour_list_x = []
    non_seizure_hour_list_x = []

    if duration_x - 1 + base_x <= seizure_list_end_x - 1 + timestep_x :
        seizure_hour_list_x = seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1), (base_x + 3600, base_x + 7200 - 1)]

    elif seizure_list_start_x - timestep_x <= base_x :
        seizure_hour_list_x = seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 7200 + 1, duration_x - 1 + base_x - 3600), (duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]

    elif duration_x - 1 + base_x  - (seizure_list_end_x - 1 + timestep_x) >= 3600 :
        seizure_hour_7200_10800, non_seizure_hour_7200_10800 = compute_duration_7200_10800(duration_x - 3600, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_7200_10800
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_7200_10800
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]

    elif seizure_list_start_x - timestep_x - 1 - base_x + 1 >= 3600 :
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        seizure_hour_7200_10800, non_seizure_hour_7200_10800 = compute_duration_7200_10800(duration_x - 3600, base_x + 3600, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_7200_10800
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_7200_10800

    return seizure_hour_list_x, non_seizure_hour_list_x
        
        
def compute_duration_14400_18000(duration_x, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x):
    assert duration_x >= 14400 and duration_x < 18000
    assert duration_x - 1 + base_x >= seizure_list_end_x - 1
    assert base_x <= seizure_list_start_x
    
    seizure_hour_list_x = []
    non_seizure_hour_list_x = []

    if duration_x - 1 + base_x <= seizure_list_end_x - 1 + timestep_x :
        seizure_hour_list_x = seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1), (base_x + 3600, base_x + 7200 - 1), (base_x + 7200, base_x + 10800 - 1)]
    
    elif seizure_list_start_x - timestep_x <= base_x :
        seizure_hour_list_x = seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 10800 + 1, duration_x - 1 + base_x - 7200),(duration_x - 1 + base_x - 7200 + 1, duration_x - 1 + base_x - 3600), (duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]

    elif duration_x - 1 + base_x  - (seizure_list_end_x - 1 + timestep_x) >= 3600 :
        seizure_hour_10800_14400, non_seizure_hour_10800_14400 = compute_duration_10800_14400(duration_x - 3600, base_x, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_10800_14400
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_10800_14400
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(duration_x - 1 + base_x - 3600 + 1, duration_x - 1 + base_x)]

    elif seizure_list_start_x - timestep_x - 1 - base_x + 1 >= 3600 :
        non_seizure_hour_list_x = non_seizure_hour_list_x + [(base_x, base_x + 3600 - 1)]
        seizure_hour_10800_14400, non_seizure_hour_10800_14400 = compute_duration_10800_14400(duration_x - 3600, base_x + 3600, seizure_list_start_x, seizure_list_end_x, timestep_x)
        seizure_hour_list_x = seizure_hour_list_x + seizure_hour_10800_14400
        non_seizure_hour_list_x = non_seizure_hour_list_x + non_seizure_hour_10800_14400

    return seizure_hour_list_x, non_seizure_hour_list_x
        
        


def func_compute_seizure_nonseizure_hour_time_records(allowed_max_nonseizure_timelenth_in_seizure_segment) :
    dir = '/home/xya238/CHB_MIT_EEGdata/'
    folder_list = ['chb01/','chb02/','chb03/','chb04/','chb05/','chb06/','chb07/','chb08/',
                   'chb09/','chb10/','chb11/','chb12/','chb13/','chb14/','chb15/','chb16/',
                   'chb17/','chb18/','chb19/','chb20/','chb21/','chb22/','chb23/','chb24/']

    more_1_hour_seizure_distance_information_dict = {}

    seizure_time_record_dict = {}

    non_seizure_time_record_dict = {}    # each non-seizure record is with the length of one hour

    for folder in folder_list:
        dir_X = dir + folder

        files = os.listdir(dir_X)
        file_name_list = list(filter(lambda x:x.endswith('.edf'),files))
        file_name_list.sort()

        seizure_time_dict = {}  # create a dictionary to record the seizure information about each edf file in the folder

        if folder != 'chb24/':
            with open(dir_X + dir_X[-6:-1] + '-summary.txt','r') as f:     # to read seizure information about each file in the folder
                line = f.readline()
        
                while line:
            
                    line = line.strip()
                    if line[0:10]=='File Name:':
                        file_name = line[11:]
                        
                        line_start_time = f.readline().strip()  #read file start time
                        record_start_time = line_start_time[line_start_time.find(':')+2 :]
                        assert line_start_time[0:line_start_time.find(':')] == 'File Start Time'
                        
                        line_end_time = f.readline().strip()   # read file end time
                        record_end_time = line_end_time[line_end_time.find(':')+2 :]
                        assert line_end_time[0:line_end_time.find(':')] == 'File End Time'
                        
                        line = f.readline().strip() # read seizure number
                
                        assert line[0:27]=='Number of Seizures in File:'
                
                        if int(line[28:])==0:
                            seizure_time_dict[file_name] = ((record_start_time, record_end_time), None)
                        else:
                            assert int(line[28:]) > 0
                            seizure_time_list = []
                            for i in range(int(line[28:])):
                                line_1 = f.readline().strip()
                                seizure_start_time = int(re.findall(r"\d+\.?\d*",line_1)[-1])
                                line_2 = f.readline().strip()
                                seizure_end_time = int(re.findall(r"\d+\.?\d*",line_2)[-1])
                                seizure_time_list.append((seizure_start_time,seizure_end_time))
                            seizure_time_dict[file_name] = ((record_start_time, record_end_time), seizure_time_list)
                    line = f.readline()
                    
        else:
            seizure_time_dict = {
                'chb24_01.edf':((), [(480,505),(2451,2476)]),
                'chb24_02.edf':((), None),
                'chb24_03.edf':((), [(231,260),(2883,2908)]),
                'chb24_04.edf':((), [(1088,1120),(1411,1438),(1745,1764)]),
                'chb24_05.edf':((), None),
                'chb24_06.edf':((), [(1229,1253)]),
                'chb24_07.edf':((), [(38,60)]),
                'chb24_08.edf':((), None),
                'chb24_09.edf':((), [(1745,1764)]),
                'chb24_10.edf':((), None),
                'chb24_11.edf':((), [(3527,3597)]),
                'chb24_12.edf':((), None),
                'chb24_13.edf':((), [(3288,3304)]),
                'chb24_14.edf':((), [(1939,1966)]),
                'chb24_15.edf':((), [(3552,3569)]),
                'chb24_16.edf':((), None),
                'chb24_17.edf':((), [(3515,3581)]),
                'chb24_18.edf':((), None),
                'chb24_19.edf':((), None),
                'chb24_20.edf':((), None),
                'chb24_21.edf':((), [(2804,2872)]),
                'chb24_22.edf':((), None)
        
                }


        non_seizure_hour_dict_a_folder = {}
        seizure_hour_dict_a_folder = {}

        for file in file_name_list :
            if file in ['chb12_27.edf','chb12_28.edf','chb12_29.edf'] :
                continue

            f = pyedflib.EdfReader(dir_X+file)
            datarecord_duration = f.getFileDuration()
            assert datarecord_duration < 18000 

            non_seizure_hour_list_a_file = []
            seizure_hour_list_a_file = []

            if seizure_time_dict[file][1] == None :
                for i in range(datarecord_duration // 3600) :
                    non_seizure_hour_list_a_file.append((i*3600,(i+1)*3600-1))
                non_seizure_hour_dict_a_folder[file] = non_seizure_hour_list_a_file
                seizure_hour_dict_a_folder[file] = seizure_hour_list_a_file
                
            else :
                seizure_time_list = seizure_time_dict[file][1]
                if len(seizure_time_list) > 1 :
                    for i in range(len(seizure_time_list)-1) :
                        assert seizure_time_list[i][0] < seizure_time_list[i+1][0]
                        assert seizure_time_list[i][1] < seizure_time_list[i+1][1]

                if seizure_time_list[-1][1] - seizure_time_list[0][0] >= 3600 - 2*allowed_max_nonseizure_timelenth_in_seizure_segment :
                    print(file, ', its seizure time length is more than 3600-2*allowed_max_nonseizure_timelenth_in_seizure_segment. We need split it manually.')
                    more_1_hour_seizure_distance_information_dict[file] = [datarecord_duration, seizure_time_list]

                    if file == 'chb06_01.edf' :
                        seizure_hour_list_a_file = [(0, 3600-1), (7200, 10800-1), (10800, 14400-1)]
                        non_seizure_hour_list_a_file = [(3600, 7200-1)]
                            
                    elif file == 'chb06_04.edf' :
                        seizure_hour_list_a_file = [(0, 3600-1), (3600, 7200-1)]
                        non_seizure_hour_list_a_file = [(7200, 10800-1)]
                            
                    elif file == 'chb09_08.edf' :
                        seizure_hour_list_a_file = [(0, 3600-1), (7200, 10800-1)]
                        non_seizure_hour_list_a_file = [(3600, 7200-1), (10800, 14400-1)]
                            
                    elif file == 'chb23_08.edf' :
                        seizure_hour_list_a_file = [(0, 3600-1), (3600, 7200-1)]
                            
                    elif file == 'chb23_09.edf' :
                        seizure_hour_list_a_file = [(0, 3600-1), (3600, 7200-1), (7200, 10800-1)]
                        non_seizure_hour_list_a_file = [(10800, 14400-1)]

                else :
                    if datarecord_duration < 3600 :
                        print(file, ' its datarecord duration is: ', datarecord_duration, ' less than 3600.')
                        seizure_hour_list_a_file = [(0, datarecord_duration - 1)]   

                    if datarecord_duration >= 3600 and datarecord_duration < 7200 :
                        seizure_hour_list_x_3600_7200, non_seizure_hour_list_x_3600_7200 = compute_duration_3600_7200(datarecord_duration, 0,
                                                                                                                      seizure_time_list[0][0],
                                                                                                                      seizure_time_list[-1][1],
                                                                                                                      allowed_max_nonseizure_timelenth_in_seizure_segment)
                        seizure_hour_list_a_file = seizure_hour_list_a_file + seizure_hour_list_x_3600_7200  
                        non_seizure_hour_list_a_file = non_seizure_hour_list_a_file + non_seizure_hour_list_x_3600_7200

                        for item in seizure_hour_list_a_file + non_seizure_hour_list_a_file :
                            assert item[1] - item[0] + 1 == 3600
                        

                    if datarecord_duration >= 7200 and datarecord_duration < 10800 :
                        seizure_hour_list_x_7200_10800, non_seizure_hour_list_x_7200_10800 = compute_duration_7200_10800(datarecord_duration, 0,
                                                                                                                         seizure_time_list[0][0],
                                                                                                                         seizure_time_list[-1][1],
                                                                                                                         allowed_max_nonseizure_timelenth_in_seizure_segment)
                        seizure_hour_list_a_file = seizure_hour_list_a_file + seizure_hour_list_x_7200_10800   
                        non_seizure_hour_list_a_file = non_seizure_hour_list_a_file + non_seizure_hour_list_x_7200_10800

                        for item in seizure_hour_list_a_file + non_seizure_hour_list_a_file :
                            assert item[1] - item[0] + 1 == 3600

                    if datarecord_duration >= 10800 and datarecord_duration < 14400 :
                        seizure_hour_list_x_10800_14400, non_seizure_hour_list_x_10800_14400 = compute_duration_10800_14400(datarecord_duration, 0,
                                                                                                                            seizure_time_list[0][0],
                                                                                                                            seizure_time_list[-1][1],
                                                                                                                            allowed_max_nonseizure_timelenth_in_seizure_segment)

                        seizure_hour_list_a_file = seizure_hour_list_a_file + seizure_hour_list_x_10800_14400    
                        non_seizure_hour_list_a_file = non_seizure_hour_list_a_file + non_seizure_hour_list_x_10800_14400

                        for item in seizure_hour_list_a_file + non_seizure_hour_list_a_file :
                            assert item[1] - item[0] + 1 == 3600

                    if datarecord_duration >= 14400 and datarecord_duration < 18000 :
                        seizure_hour_list_x_14400_18000, non_seizure_hour_list_x_14400_18000 = compute_duration_14400_18000(datarecord_duration, 0,
                                                                                                                            seizure_time_list[0][0],
                                                                                                                            seizure_time_list[-1][1],
                                                                                                                            allowed_max_nonseizure_timelenth_in_seizure_segment)

                        seizure_hour_list_a_file = seizure_hour_list_a_file + seizure_hour_list_x_14400_18000    
                        non_seizure_hour_list_a_file = non_seizure_hour_list_a_file + non_seizure_hour_list_x_14400_18000

                        for item in seizure_hour_list_a_file + non_seizure_hour_list_a_file :
                            assert item[1] - item[0] + 1 == 3600
                            

                assert seizure_hour_list_a_file != []   
                seizure_hour_dict_a_folder[file] = seizure_hour_list_a_file
                
                non_seizure_hour_dict_a_folder[file] = non_seizure_hour_list_a_file


                for seizure_item in seizure_time_list :   
                    flag = False
                    for seizure_hour_item in seizure_hour_list_a_file :
                        if seizure_item[0] >= seizure_hour_item[0] and seizure_item[1] - 1 <= seizure_hour_item[1] and seizure_item[0] <= seizure_item[1] - 1 :
                            flag = True
                            break

                    if flag == False :
                        print(file, 'the seizure duration ', seizure_item, 'in not included in a seizure_hour_record!', 'File duration: ', datarecord_duration)

                if len(seizure_hour_list_a_file) > 1 :   
                    for i in range(len(seizure_hour_list_a_file) - 1) :
                        set_A = set(seizure_hour_list_a_file[i+1 : ])
                        set_i_record = set(list(range(seizure_hour_list_a_file[i][0], seizure_hour_list_a_file[i][1]+1)))

                        for record_x in set_A :
                            set_x = set(list(range(record_x[0], record_x[1] + 1)))
                            assert set_i_record & set_x == set()

                for seizure_hour_item in seizure_hour_list_a_file :  
                    set_1 = set(list(range(seizure_hour_item[0], seizure_hour_item[1] + 1)))
                    for non_seizure_hour_item in non_seizure_hour_list_a_file :
                        set_2 = set(list(range(non_seizure_hour_item[0], non_seizure_hour_item[1] + 1)))

                        if set_1 & set_2 != set() :
                            print(file, 'seizure hour record ', seizure_hour_item, ' has common elements with non-seizure hour record ', non_seizure_hour_item)

                if len(non_seizure_hour_list_a_file) > 1 :   
                    for j in range(len(non_seizure_hour_list_a_file) - 1) :
                        set_B = set(non_seizure_hour_list_a_file[j+1 : ])
                        set_j_record = set(list(range(non_seizure_hour_list_a_file[j][0], non_seizure_hour_list_a_file[j][1]+1)))

                        for record_y in set_B :
                            set_y = set(list(range(record_y[0], record_y[1] + 1)))
                            assert set_j_record & set_y == set()
                            

        seizure_time_record_dict[folder] = seizure_hour_dict_a_folder
        non_seizure_time_record_dict[folder] = non_seizure_hour_dict_a_folder

    return seizure_time_record_dict, non_seizure_time_record_dict









              
                    
                    



            
        
