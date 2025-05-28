import os
import pyedflib
import numpy

def get_labels_excluding_three_files():
    dir = '/home/xya238/CHB_MIT_EEGdata/'

    folder_list = ['chb01/','chb02/','chb03/','chb04/','chb05/','chb06/','chb07/','chb08/',
                   'chb09/','chb10/','chb11/','chb12/','chb13/','chb14/','chb15/','chb16/',
                   'chb17/','chb18/','chb19/','chb20/','chb21/','chb22/','chb23/','chb24/']

    one_file_path = dir + 'chb01/chb01_01.edf'

    f_1 = pyedflib.EdfReader(one_file_path)
    signal_headers_1 = f_1.getSignalHeaders()

    label_set = set()
    for i in range(len(signal_headers_1)):
        label_set.add(signal_headers_1[i]['label'])

    f_1._close()
    
    for folder in folder_list:
        files = os.listdir(dir+folder)
        file_name_list = list(filter(lambda x:x.endswith('.edf'),files))
        file_name_list.sort()

        for file in file_name_list:
            if file == 'chb12_27.edf' or file == 'chb12_28.edf' or file == 'chb12_29.edf':
                continue
            f = pyedflib.EdfReader(dir+folder+file)
            signal_headers = f.getSignalHeaders()
            label_set_X = set()

            for j in range(len(signal_headers)):
                label_set_X.add(signal_headers[j]['label'])

            label_set = label_set & label_set_X

    label_list = []

    for x in label_set:
        label_list.append(x)

    label_list.sort()  

    return label_list
