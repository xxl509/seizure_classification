import os
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import itertools

import gc
from random import shuffle

from scipy import stats


def get_hour_sample_data(x_test_folder, x_validation_data_ratio, x_majority, x_time_length_segment):

    # data path
    path = '/scratch/xya238/CHB_MIT_process_results/hour_data_samples_overlap_6s_seg_4s_major_fullsei_start_end_pre_post_nonsei/'

    with_sei_files = list(filter(lambda x: x.endswith('_with_seizure.npy'),os.listdir(path)))
    
    #with_sei_files.sort()

    train_validate_data_files = []
    test_data_files = []
    for x_file in with_sei_files :
        if x_file.find(x_test_folder) != -1 :
            test_data_files.append(x_file)
        else :
            train_validate_data_files.append(x_file)

    train_validate_data_files.sort()

    assert len(train_validate_data_files) + len(test_data_files) == len(with_sei_files)

    #__________________________________________________________________________________________________________________
    # to achieve test_data and test_label
    #_________________________________________________________________________________________________________________
    
    test_data = []

    for z_file in test_data_files :
        z_sample = np.load(path + z_file)[0]
        for i in range(len(z_sample)) :
            assert len(set(z_sample[i, -1])) == 1
            assert list(set(z_sample[i, -1]))[0][0:5] == x_test_folder

        test_data.append(z_sample[:, 0:-1])

    del z_sample
    gc.collect()

    test_data = np.array(test_data)
    assert len(test_data.shape) == 4

    test_label = []

    sei_start_label_for_seizure = set(['seizure_start_' + str(x) + 's' for x in range(x_majority, x_time_length_segment + 1)])
    sei_end_label_for_seizure = set(['seizure_end_' + str(x) + 's' for x in range(x_majority, x_time_length_segment + 1)])
    sei_start_end_label_for_seizure = sei_start_label_for_seizure | sei_end_label_for_seizure

    for i in range(len(test_data)) :
        z_sample_label = []
        for j in range(test_data.shape[1]) :
            label_set_one_segment = set(test_data[i, j, -1])
            assert len(label_set_one_segment) == 1
            assert 'non_seizure' not in label_set_one_segment
            
            if label_set_one_segment == {'full_seizure'} :
                z_sample_label.append(1)
            elif label_set_one_segment.issubset(sei_start_end_label_for_seizure) :
                z_sample_label.append(1)
            else :
                z_sample_label.append(0)

        test_label.append(z_sample_label)

    del z_sample_label
    gc.collect()

    test_label = np.array(test_label).astype('int32')
    test_data = test_data[:, :, 0:-1].astype('float32')

    assert test_label.shape[0] == test_data.shape[0] and test_label.shape[1] == test_data.shape[1]
        

    #__________________________________________________________________________________________________________________
    # to achieve validate_data, validate_label, train_data, train_label
    #_________________________________________________________________________________________________________________
      
    train_validate_data = []
    for y_file in train_validate_data_files :
        y_data = np.load(path + y_file)[0]
        train_validate_data.append(y_data)

    del y_data
    gc.collect()

    num_validate_samples = int(x_validation_data_ratio * len(train_validate_data))

    validate_data = train_validate_data[(-1) * num_validate_samples :]
    
    for i in range(1, num_validate_samples + 1) :
        train_validate_data.pop((-1)*i)
        
    train_data = train_validate_data

    del train_validate_data
    gc.collect()

    validate_data = np.array(validate_data)[:, :, 0:-1]
    train_data = np.array(train_data)[:, :, 0:-1]

    assert len(validate_data.shape) == 4 and len(train_data.shape) == 4


    #__________________________________________________________________________________________________________________
    # to achieve validate_data and validate_label
    #_________________________________________________________________________________________________________________
            
    validate_label = []
    for i in range(len(validate_data)) :
        x_sample_label = []
        for j in range(validate_data.shape[1]) :
            label_set_one_segment = set(validate_data[i,j,-1])
            assert len(label_set_one_segment) == 1
            assert 'non_seizure' not in label_set_one_segment
            
            if label_set_one_segment == {'full_seizure'} :
                x_sample_label.append(1)
            elif label_set_one_segment.issubset(sei_start_end_label_for_seizure) :
                x_sample_label.append(1)
            else :
                x_sample_label.append(0)

        validate_label.append(x_sample_label)

    del x_sample_label
    gc.collect()

    validate_label = np.array(validate_label).astype('int32')
    validate_data = validate_data[:, :, 0:-1].astype('float32')

    assert validate_label.shape[0] == validate_data.shape[0] and validate_label.shape[1] == validate_data.shape[1]


    #__________________________________________________________________________________________________________________
    # to achieve train_data and train_label
    #_________________________________________________________________________________________________________________
        
    train_label = []
    for i in range(len(train_data)) :
        y_sample_label = []
        for j in range(train_data.shape[1]) :
            label_set_one_segment = set(train_data[i, j, -1])
            assert len(label_set_one_segment) == 1
            assert 'non_seizure' not in label_set_one_segment

            if label_set_one_segment == {'full_seizure'} :
                y_sample_label.append(1)
            elif label_set_one_segment.issubset(sei_start_end_label_for_seizure) :
                y_sample_label.append(1)
            else :
                y_sample_label.append(0)
                
        train_label.append(y_sample_label)

    del y_sample_label
    gc.collect()

    train_label = np.array(train_label).astype('int32')
    train_data = train_data[:, :, 0:-1].astype('float32')
    
    assert train_label.shape[0] == train_data.shape[0] and train_label.shape[1] == train_data.shape[1]
    assert len(train_data.shape) == 4 and len(train_label.shape) == 2
    assert len(validate_data.shape) == 4 and len(validate_label.shape) == 2
    assert len(test_data.shape) == 4 and len(test_label.shape) == 2
                       
    return train_data, train_label, test_data, test_label, validate_data, validate_label

    

def get_training_set_with_four_dimension(inputs, labels, batch_size_train):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    return dataset.repeat().batch(batch_size_train)

def get_bn_stats_set_with_four_dimension(inputs, labels, batch_size_bn_stats):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    return dataset.repeat().batch(batch_size_bn_stats)

def get_prediction_set_with_four_dimension(inputs, labels, batch_size_valid):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset.batch(batch_size_valid)


def get_iterators_with_four_dimension(session, handle, X_train, Y_train, X_test, Y_test, X_validate, Y_validate,
                                      batch_size_train, batch_size_bn_stats, batch_size_valid):

    _, num_segments, time_steps, chans = X_train.shape
    
    inputs_ph = tf.placeholder('float32', [None, num_segments, time_steps, chans],
                               name="all_inputs")
    labels_ph = tf.placeholder('int32', [None, num_segments],
                               name="all_labels")

    training_dataset = get_training_set_with_four_dimension(inputs_ph, labels_ph, batch_size_train)
    bn_stats_dataset = get_bn_stats_set_with_four_dimension(inputs_ph, labels_ph, batch_size_bn_stats)
    validation_dataset = get_prediction_set_with_four_dimension(inputs_ph, labels_ph, batch_size_valid)
    test_dataset = get_prediction_set_with_four_dimension(inputs_ph, labels_ph, batch_size_valid)

    training_iterator = training_dataset.make_initializable_iterator()
    bn_stats_iterator = bn_stats_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Initialize iterators with their corresponding datasets
    session.run(training_iterator.initializer, feed_dict={
        inputs_ph: X_train,
        labels_ph: Y_train})
    session.run(bn_stats_iterator.initializer, feed_dict={
        inputs_ph: X_train,
        labels_ph: Y_train})
    session.run(test_iterator.initializer, feed_dict={
        inputs_ph: X_test,
        labels_ph: Y_test})


    def init_validation_set():
        session.run(validation_iterator.initializer, feed_dict={
            inputs_ph: X_validate,
            labels_ph: Y_validate})


    # Generate handles for each iterator. 
    handles = {
        'train': session.run(training_iterator.string_handle()),
        'bn_stats': session.run(bn_stats_iterator.string_handle()),
        'validation': session.run(validation_iterator.string_handle()),
        'test': session.run(test_iterator.string_handle())
    }
    
    
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)

    return iterator, handles, init_validation_set









    
