import itertools
import os
from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

from ind_rnn_cell import IndRNNCell

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math

import gc
from random import shuffle
from scipy import stats

from load_hour_sample_data_v5 import *     

flags = tf.app.flags
FLAGS = flags.FLAGS



flags.DEFINE_integer('num_rnn_units_block_1',80,'the first number of hidden states used in the IndRNN cell')   
flags.DEFINE_integer('num_rnn_units_block_2',120,'the second number of hidden states used in the IndRNN cell')
flags.DEFINE_integer('num_rnn_units_block_3',160,'the third number of hidden states used in the IndRNN cell')
flags.DEFINE_integer('num_units_lstm',80,'the number of hidden states used in the LSTM cell')

flags.DEFINE_integer('num_fc_units_1',50,'the number of hidden states used in the fully connected layer')

flags.DEFINE_string('activation_func_fully_connect', 'None', 'the activation function in the first fully connected layer') 

flags.DEFINE_float('learning_rate_init',0.001,'the initial value of learning rate')
flags.DEFINE_integer('learning_rate_decay_steps',600000,'a reference step number to decrease learning rate')

flags.DEFINE_float('loss_coefficient', 0.1, 'the weight used to caculate the loss between prediction and target')
flags.DEFINE_float('weight_decay', 0.01, 'the weight used to caculate the trainable-variable loss') 
flags.DEFINE_float('initial_bias', 0.001, 'the initial value of bias')

flags.DEFINE_bool('clip_gradients',True,'to mark whether to clip gradients')
flags.DEFINE_integer('batch_size_train',2,'the batch size when training')
flags.DEFINE_integer('epochs', 30, 'the number of training epochs')

flags.DEFINE_integer('batch_size_bn_stats',24,'the batch size when computing the population statistics for batch normalization')
flags.DEFINE_integer('batch_size_valid',2,'the batch size when validating and testing')

flags.DEFINE_bool('normalize',False,'whether to normalize the training dat, validation data and test data')

flags.DEFINE_bool('single_attention_vector',True,'whether attention weights will be shared over time steps for each sample')

flags.DEFINE_integer('num_layers_in_block', 3, 'the number of layers in each block')
flags.DEFINE_integer('num_blocks', 3, 'the number of all the blocks')

flags.DEFINE_string('test_folder', 'chb06', 'the name of folder for test data')   # input testing patient
flags.DEFINE_float('validate_data_ratio', 0.1, 'the ratio of validate data in train-validate data')


#*************************************************************************************************
# load data
#**************************************************************************************************
data_majority_for_seizure = 4   
time_length_segment = 6   

X_train, Y_train, X_test, Y_test, X_validate, Y_validate = get_hour_sample_data(FLAGS.test_folder, FLAGS.validate_data_ratio, data_majority_for_seizure, time_length_segment) 


num_training_samples, num_segments, time_steps, chans = X_train.shape

if FLAGS.normalize == True :   
    X_train = X_train.reshape([-1, time_steps, chans])
    X_validate = X_validate.reshape([-1, time_steps, chans])
    X_test = X_test.reshape([-1, time_steps, chans])
    
    X_train = stats.zscore(X_train, axis=0, ddof=0)
    X_validate = stats.zscore(X_validate, axis=0, ddof=0)
    X_test = stats.zscore(X_test, axis=0, ddof=0)

    X_train = X_train.astype('float32')
    X_validate = X_validate.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.reshape([-1, num_segments, time_steps, chans])
    X_validate = X_validate.reshape([-1, num_segments, time_steps, chans])
    X_test = X_test.reshape([-1, num_segments, time_steps, chans])

print('X_train:',type(X_train),X_train.shape,X_train.dtype)
print('Y_train:',type(Y_train),Y_train.shape,Y_train.dtype)
print('X_test:',X_test.shape)
print('Y_test:',Y_test.shape)
print('X_validate:',X_validate.shape)
print('Y_validate:',Y_validate.shape)


recurrent_max = pow(2, 1 / time_steps)
last_layer_lower_bound = pow(0.5, 1 / time_steps)

num_classes = 2

phase_train = 'train'
phase_bn_stats = 'bn_stats'
phase_valid = 'validation'
phase_test = 'test'

out_dir = 'out/%s/' % datetime.utcnow()
save_path = out_dir + 'model.ckpt'

print('time steps:',time_steps)
print('chans:',chans)
print('recurrent_max:',recurrent_max)
print('last_layer_lower_bound:',last_layer_lower_bound)


#*************************************************************************************************
# build model 
#*************************************************************************************************

def tf_attention_3d_block(inputs) :   
    _, dim_1, dim_2 = inputs.get_shape().as_list()
    a = tf.layers.dense(inputs,
                        units=dim_2,
                        activation=tf.nn.softmax,
                        use_bias=True,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        trainable=True,
                        name='attention_dense',
                        reuse=None
                        )

    if FLAGS.single_attention_vector :
        a = tf.reduce_mean(a,axis=1,keepdims=True)
        a = tf.tile(a,multiples=[1,dim_1,1])

    output_attention_multiply = tf.multiply(inputs,a)
        
    return output_attention_multiply



def build_block_IndRNN(inputs, phase, block_id, num_rnn_units) :
    layer_input = inputs
    layer_output = None
    input_init = tf.random_uniform_initializer(-0.001, 0.001)

    for layer in range(1, FLAGS.num_layers_in_block + 1) :
        if layer < FLAGS.num_layers_in_block or block_id < FLAGS.num_blocks :
            recurrent_init_lower = 0
        else :
            recurrent_init_lower = last_layer_lower_bound

        recurrent_init = tf.random_uniform_initializer(recurrent_init_lower,
                                                       recurrent_max)

        cell = IndRNNCell(num_rnn_units,
                          recurrent_max_abs=recurrent_max,
                          input_kernel_initializer=input_init,
                          recurrent_kernel_initializer=recurrent_init)

        layer_output, _ = tf.nn.dynamic_rnn(cell, layer_input,
                                            dtype=tf.float32,
                                            scope='block{0}_layer{1}'.format(block_id, layer))

        is_training = tf.logical_or(tf.equal(phase, phase_train),
                                    tf.equal(phase, phase_bn_stats))

        layer_output = tf.layers.batch_normalization(layer_output,
                                                     training=is_training,
                                                     momentum=0)

        def update_population_stats():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                return tf.identity(layer_output)


        layer_output = tf.cond(tf.equal(phase, phase_bn_stats),
                               true_fn = update_population_stats,
                               false_fn = lambda: layer_output)

        layer_input = tf.concat([layer_input, layer_output], axis=2)

    block_output = layer_output
    
    return block_output 


#***********************************************************************************************

def build(inputs, labels, phase):

    inputs_3d = tf.reshape(inputs, [-1, time_steps, chans], name='reshape_inputs_from_4d_to_3d')
     
    inputs_with_attention = tf_attention_3d_block(inputs_3d)   # add attention onto each channel

    block_input = inputs_with_attention

    num_rnn_units_in_block_list = [FLAGS.num_rnn_units_block_1, FLAGS.num_rnn_units_block_2, FLAGS.num_rnn_units_block_3]

    assert FLAGS.num_blocks == len(num_rnn_units_in_block_list)
    
    for i in range(1, FLAGS.num_blocks + 1) :
        x_num_rnn_units_block = num_rnn_units_in_block_list[i-1]

        block_output = build_block_IndRNN(block_input, phase=phase, block_id=i, num_rnn_units=x_num_rnn_units_block)

        block_input = tf.nn.pool(input=block_output,window_shape=[2],pooling_type='MAX',padding='SAME',strides=[2])  

    output_average_timesteps = tf.reduce_mean(block_input,axis=1)

    _, last_num_rnn_units = output_average_timesteps.get_shape().as_list()

    assert last_num_rnn_units == num_rnn_units_in_block_list[-1]

    output_sample_form_IndRNN = tf.reshape(output_average_timesteps, [-1, num_segments, last_num_rnn_units], name='reshape_output_from_2d_to_3d')

    output_sample_form_IndRNN = tf.transpose(output_sample_form_IndRNN, [1,0,2])

    lstm_fused_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=FLAGS.num_units_lstm)
    
    output_fused_lstm, _ = lstm_fused_cell(output_sample_form_IndRNN, dtype=tf.float32) 

    output_fused_lstm = tf.transpose(output_fused_lstm, [1, 0, 2])

    print('output_fused_lstm shape after transpose: ', output_fused_lstm.get_shape().as_list())

    output_lstm_2d = tf.reshape(output_fused_lstm, [-1, FLAGS.num_units_lstm], name='reshape_lstm_output_from_3d_to_2d')


       
    if FLAGS.activation_func_fully_connect == 'tanh' :
        X_activation_func = tf.nn.tanh
    elif FLAGS.activation_func_fully_connect == 'relu' :
        X_activation_func = tf.nn.relu
    elif FLAGS.activation_func_fully_connect == 'None' :
        X_activation_func = None
        
   
    logits_1 = tf.contrib.layers.fully_connected(
        inputs=output_lstm_2d,
        num_outputs=FLAGS.num_fc_units_1,
        activation_fn=X_activation_func,              
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.constant_initializer(FLAGS.initial_bias),
        biases_regularizer=None,
        variables_collections=None,
        trainable=True,
        scope='fully_connected_layer_1'
        )


    logits_final = tf.contrib.layers.fully_connected(
        inputs=logits_1,
        num_outputs=num_classes,
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.constant_initializer(FLAGS.initial_bias),
        biases_regularizer=None,
        variables_collections=None,
        trainable=True,
        scope='fully_connected_layer_2'
        )


    logits_final = tf.reshape(logits_final, [-1, num_segments, num_classes])

    loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits_final,[-1, num_classes]), labels=tf.reshape(labels,[-1])))

    loss_1 = loss_1 * num_segments  

    loss_2 =  tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    loss = FLAGS.loss_coefficient * loss_1 + FLAGS.weight_decay * loss_2

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_init, global_step,
                                               FLAGS.learning_rate_decay_steps, 0.1,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    if FLAGS.clip_gradients :
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables))
    else:
        optimize = optimizer.minimize(loss, global_step=global_step)

    correct_pred = tf.equal(tf.argmax(logits_final, 2, output_type=tf.int32), labels)
    acc_each_sample = tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=1)  
    accuracy = tf.reduce_mean(acc_each_sample)  # It means the average accuracy over samples in a batch

    pred = tf.argmax(logits_final, 2, output_type=tf.int32)   

    softmax_probability = tf.nn.softmax(tf.reshape(logits_final, [-1, num_classes]))

    softmax_probability = tf.reshape(softmax_probability, [-1, num_segments, num_classes])
    
  
    return loss, accuracy, optimize, pred, labels, softmax_probability



#*************************************************************************************************

def compute_cm_2(confu_mat,num_classes=2):  # metric-computing way corresponding to the case of 2 classes
    TP = confu_mat[1,1]
    FP = confu_mat[0,1]
    FN = confu_mat[1,0]
    TN = confu_mat[0,0]

    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)
    F1 = 2*TP / (2*TP+FP+FN)
    accuracy = (TP+TN) / (TP+FN+TN+FP)
    precision = TP / (TP+FP)
    return sensitivity, specificity, F1, precision, accuracy



def evaluate(session, loss_op, accuracy_op, pred_op, labels_op, softmax_prob_op, feed_dict):

    loss_list, pred_list, true_labels_list, softmax_prob_list = [], [], [], []

    while True:
        try:
            loss, pred, true_labels, softmax_prob = session.run(
                [loss_op, pred_op, labels_op, softmax_prob_op],
                feed_dict=feed_dict)

            loss_list.append(loss)
            pred_list.append(pred)
            true_labels_list.append(true_labels)
            softmax_prob_list.append(softmax_prob)
        except tf.errors.OutOfRangeError:
            break

    pred_all = np.concatenate(pred_list,axis=0)
    true_labels_all = np.concatenate(true_labels_list,axis=0)
    softmax_prob_all = np.concatenate(softmax_prob_list, axis=0)

    assert pred_all.shape == true_labels_all.shape
    assert true_labels_all.shape[0] == softmax_prob_all.shape[0]
    
    cm_list = []
    sensi_list = []
    spec_list = []
    F1_list = []
    prec_list = []
    acc_list = []
    
    for i in range(len(pred_all)) :
        cm_x = confusion_matrix(true_labels_all[i], pred_all[i])
        cm_list.append(cm_x)
        sensitivity_x, specificity_x, F1_x, precision_x, accuracy_x = compute_cm_2(cm_x)
        sensi_list.append(sensitivity_x)
        spec_list.append(specificity_x)
        F1_list.append(F1_x)
        prec_list.append(precision_x)
        acc_list.append(accuracy_x)
        
  
    return np.mean(loss_list), cm_list, np.mean(sensi_list), np.mean(spec_list), np.mean(F1_list), np.mean(prec_list), np.mean(acc_list), pred_all, true_labels_all, softmax_prob_all


#********************************************************************************************************************************

sess = tf.Session(graph=tf.get_default_graph())

data_handle = tf.placeholder(tf.string, shape=[], name="data_handle")

iterator, handles, init_validation_set = get_iterators_with_four_dimension(sess, data_handle, X_train, Y_train, X_test, Y_test, X_validate, Y_validate,
                                                       FLAGS.batch_size_train, FLAGS.batch_size_bn_stats, FLAGS.batch_size_valid)

# inputs and labels can contain data from any of the datasets
inputs_4d, labels_2d = iterator.get_next()

phase = tf.placeholder(tf.string, shape=[], name="phase")
loss_op, accuracy_op, train_op, pred_op, labels_op, softmax_prob_op = build(inputs_4d, labels_2d, phase)

# Train the model
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print('size of tf graph: ', sess.graph_def.ByteSize()/(1024**3))

train_losses = []
train_accuracies = []

for step in itertools.count():
    loss, accuracy, _ = sess.run(
        [loss_op, accuracy_op, train_op],
        feed_dict={data_handle: handles[phase_train], phase: phase_train})
    train_losses.append(loss)
    train_accuracies.append(accuracy)

    if step % 1 == 0 :
        print("{}, Step: {}, Loss: {:.4f}, Acc: {:.4f}".format(
            datetime.utcnow(), step + 1, np.mean(train_losses),
            np.mean(train_accuracies)))
        train_losses.clear()
        train_accuracies.clear()

    if (step + 1) * FLAGS.batch_size_train >= FLAGS.epochs * num_training_samples :
        # Save the model to disk
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_sess_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % save_sess_path)

    if (step + 1) % (num_training_samples // FLAGS.batch_size_train) == 0 or (step + 1) * FLAGS.batch_size_train >= FLAGS.epochs * num_training_samples :
        sess.run([loss_op], feed_dict={
            data_handle: handles[phase_bn_stats],
            phase: phase_bn_stats})

        init_validation_set()
        feed_dict = {data_handle: handles[phase_valid], phase: phase_valid}
        loss_val, _, sensitivity_val, specificity_val, F1_val, precision_val, accuracy_val, _, _, _ = \
                  evaluate(sess, loss_op, accuracy_op, pred_op, labels_op, softmax_prob_op, feed_dict)
        
        print("{}, Step: {}, valid_ave_loss: {:.4f}, valid_ave_sensitivity: {:.4f}, valid_ave_specificity: {:.4f}, valid_ave_F1: {:.4f}, valid_ave_precision: {:.4f}, valid_ave_accuracy: {:.4f}".format(
            datetime.utcnow(),
            step + 1,
            loss_val,
            sensitivity_val,
            specificity_val,
            F1_val,
            precision_val,
            accuracy_val
            ))
        
        if (step + 1) * FLAGS.batch_size_train >= FLAGS.epochs * num_training_samples : 
            # Run the final test
            print('Epoch when testing: ', (step + 1) * FLAGS.batch_size_train / num_training_samples)
            
            feed_dict = {data_handle: handles[phase_test], phase: phase_test}
            loss_test, cm_test, sensitivity_test, specificity_test, F1_test, precision_test, accuracy_test, pred_test, true_labels_test, softmax_prob_test = \
                       evaluate(sess, loss_op, accuracy_op, pred_op, labels_op, softmax_prob_op, feed_dict)
            
            print("{}, Step: {}, test_loss: {:.4f}, test_cm: {}, test_ave_sensitivity: {:.4f}, test_ave_specificity: {:.4f}, test_ave_F1: {:.4f}, test_ave_precision: {:.4f}, test_ave_acc: {:.4f}".format(
                datetime.utcnow(),
                step + 1,
                loss_test,
                cm_test,
                sensitivity_test,
                specificity_test,
                F1_test,
                precision_test,
                accuracy_test
                ))

            pred_test = np.expand_dims(pred_test, axis=2)
            true_labels_test = np.expand_dims(true_labels_test, axis=2)
            pred_and_true_labels = np.concatenate([pred_test, true_labels_test], axis=2)

            title_pred_true_label = np.array([['Predicted_label', 'Truth_label']])
            pred_and_true_labels = list(pred_and_true_labels)
            for i in range(len(pred_and_true_labels)) :
                pred_and_true_labels[i] = np.concatenate([title_pred_true_label, pred_and_true_labels[i]], axis=0)

            pred_and_true_labels = np.array(pred_and_true_labels)
            pred_and_true_labels = np.reshape(pred_and_true_labels, [-1,2]) 

            pd.DataFrame(pred_and_true_labels).to_csv('ADIR_LSTM_test_' + FLAGS.test_folder + '_pred_and_truth_csv.csv', header=None, index=None)

            softmax_prob_test = list(softmax_prob_test)
            title_prob = np.array([['Non_seizure_probability', 'Seizure_probability']])
            for j in range(len(softmax_prob_test)) :
                softmax_prob_test[j] = np.concatenate([title_prob, softmax_prob_test[j]], axis=0)

            softmax_prob_test = np.array(softmax_prob_test)
            softmax_prob_test = np.reshape(softmax_prob_test,[-1,2])
            pd.DataFrame(softmax_prob_test).to_csv('ADIR_LSTM_test_' + FLAGS.test_folder + '_probability_csv.csv', header=None, index=None)
                                   
            break
            
                 
                                                                       
           
















        



