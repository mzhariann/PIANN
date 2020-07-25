from utils import *
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from combi_model_11 import CombiModel11

seed_value= 12321
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

args = parse_args()
print(args)
#data_path = 'data/favorita_360_shuffled.csv'
#data_path = 'data/rossman_365.csv'
#data_path = 'data/ah_online_sales_2030_764.csv'
data_path = 'data/dunnhumpy.csv'
save_path = 'result_simple_earlystopping_'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/pakdd/favorita/'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/pakdd/kernel/'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/pakdd/walmart/'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/rossman/result_'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/ah/result_'+args.job_id+'/'
#save_path = '/ivi/ilps/personal/mariann/pakdd/dh/result_'+args.job_id+'/'

pred_steps = args.pred_steps
batch_size = args.batch_size
epochs = 1000

if not os.path.exists(save_path):
    os.makedirs(save_path)

df = pd.read_csv(data_path)
#df = df[:100]

train_enc_start, train_enc_end, train_pred_start, train_pred_end, val_enc_start, val_enc_end, \
           val_pred_start, val_pred_end = get_dates(df, pred_steps)
date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                          data=[i for i in range(len(df.columns[1:]))])
series_array = df[df.columns[1:]].values

simple_model = CombiModel11(args,seed_value)
simple_model.model.compile(Adam(), loss='mean_absolute_error')

print(simple_model.model.summary())
print("Model compiled")

encoder_input_data, decoder_target_data, _, _ =\
    simple_model.get_train_data(series_array, date_to_index, train_enc_start, train_enc_end, train_pred_start, train_pred_end)
print("Train data loaded")
print(encoder_input_data[0].shape)
print(encoder_input_data[1].shape)
print(decoder_target_data.shape)

bst_model_path = save_path+"bestmodel.hdf5"
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

history = simple_model.model.fit(encoder_input_data, decoder_target_data, batch_size=batch_size, epochs=epochs,
                                 validation_split=0.2,callbacks=[early_stopping, model_checkpoint])

print("Training completed")
'''
bst_model_path = save_path+"bestmodel.hdf5"
simple_model.model.load_weights(bst_model_path)

encoder_input_data, decoder_target_data, encode_series_mean, encode_series_std = \
    simple_model.get_test_data(series_array, date_to_index, val_enc_start, val_enc_end, val_pred_start, val_pred_end)
print("Test data loaded")
print(encoder_input_data[0].shape)

print(encoder_input_data[1].shape)

simple_model.predict_all(encoder_input_data, decoder_target_data, save_path, df, encode_series_mean,
                                   encode_series_std)
'''