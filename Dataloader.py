import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf


def PullFromDB(event, db_file, E_threshold=0):
#    events = batch['event_no'].reset_index(drop = True)
    with sqlite3.connect(db_file) as con:
        query = 'select event_no, dom_x, dom_y, dom_z, charge_log10, time from features where event_no ==' + str(event)
        features = pd.read_sql(query, con)
        query = 'select stopped_muon from truth where event_no == ' + str(event)
        truth = pd.read_sql(query, con)
    features = features[features['charge_log10'].abs()>=E_threshold]
    
    return features, truth

#####################################################################################
def TransformFeaturesToInput(features):
    
    feat = np.array(features.drop(columns='event_no'))
    feat = np.reshape(feat, (1, feat.shape[0], feat.shape[1]))
    return feat

    # the below code isn't needed if all pulses belong to the same event.
    #tf_obj = tf.convert_to_tensor(np.array(features.drop(columns='event_no')))
    #tf_obj_num = tf.convert_to_tensor(np.array(features['event_no']))
    #_, _, counts = tf.unique_with_counts(tf_obj_num) # find batches
    #max_length = np.max(counts)
    #feat = np.split(tf_obj, counts) # batched features (with event_no)
    
    #return feat
    

#####################################################################################
def TransformTruthToInput(truth, N_categories=2):
    targ = truth.values        
    targ2 = tf.keras.utils.to_categorical(targ, N_categories)
    
    if N_categories == 1:
        out = targ
    else:
        out = targ2
    
    return out
