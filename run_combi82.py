from utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from my_model_h522 import MyModelH522
from combi_model_44 import CombiModel44
from combi_model_82 import CombiModel82
from combi_model_11 import CombiModel11
from keras.optimizers import Adam

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
data_path = 'data/favorita_360_shuffled.csv'
#save_path = '/ivi/ilps/personal/mariann/pakdd/favorita/'+args.job_id+'/'
save_path = 'result_combi82'+args.job_id+'/'

pred_steps = args.pred_steps #1#8
batch_size = args.batch_size #2**5
epochs = 1000

if not os.path.exists(save_path):
    os.makedirs(save_path)

df = pd.read_csv(data_path)
df = df[:100]

train_enc_start, train_enc_end, train_pred_start, train_pred_end, val_enc_start, val_enc_end, \
           val_pred_start, val_pred_end = get_dates(df, pred_steps)
date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                          data=[i for i in range(len(df.columns[1:]))])
series_array = df[df.columns[1:]].values
exog_array = get_exog_array_other_stores_with_item_str(df,'data/all_other_stores/',54)


h522_model = MyModelH522(exog_array,args)
combi11 = CombiModel11(args,seed_value)
#model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi11_delta2_hypertune/combi11_delta2_146033_79/bestmodel.hdf5'
#model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi11_delta8_hypertune/combi11_delta8_146076_462/bestmodel.hdf5'
#model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi11_delta16_hypertune/combi11_delta16_146093_141/bestmodel.hdf5'
model_path = 'vis_data/combi11_138799_10/bestmodel.hdf5'
combi11.model.load_weights(model_path)

teacher_encoder_input_data, _, _, _ =\
    combi11.get_train_data(series_array, date_to_index, train_enc_start, train_enc_end, train_pred_start, train_pred_end)
wavenet_out = combi11.model.predict(teacher_encoder_input_data)

teacher = CombiModel44(exog_array,args,h522_model.model,seed_value)
model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi44_delta2_hypertune/combi44_delta2_157444_17/bestmodel.hdf5'
#model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi44_delta8_hypertune/combi44_delta8_157544_17/bestmodel.hdf5'
#model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/combi44_delta16_hypertune/combi44_delta16_157602_17/bestmodel.hdf5'
teacher.model.load_weights(model_path)

teacher_encoder_input_data, _, _, _ =\
    teacher.get_train_data(series_array, date_to_index, train_enc_start, train_enc_end, train_pred_start, train_pred_end)
add_layer = Model(inputs=teacher.model.input, outputs = teacher.model.get_layer('add_wavenet_ffe').output)
add_layer_out = add_layer.predict([wavenet_out,teacher_encoder_input_data[1],teacher_encoder_input_data[2]])

simple_model = CombiModel82(args,seed_value,teacher.model)
simple_model.model.compile(optimizer=Adam(), loss='mean_absolute_error')

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


history = simple_model.model.fit(wavenet_out, add_layer_out, batch_size=batch_size, epochs=epochs,
                                 validation_split=0.2,callbacks=[early_stopping, model_checkpoint])

plot_history(save_path,history,'loss','val_loss')
print("Training completed")

bst_model_path = save_path+"bestmodel.hdf5"
simple_model.model.load_weights(bst_model_path)

encoder_input_data, decoder_target_data, encode_series_mean, encode_series_std = \
    simple_model.get_test_data(series_array, date_to_index, val_enc_start, val_enc_end, val_pred_start, val_pred_end)
print("Test data loaded")
print(encoder_input_data[0].shape)
print(encoder_input_data[1].shape)

combi11_encoder_input_data,_,_,_ = \
    combi11.get_test_data(series_array, date_to_index, val_enc_start, val_enc_end, val_pred_start, val_pred_end)
wavenet_out = combi11.model.predict(combi11_encoder_input_data)

simple_model.predict_all(wavenet_out, decoder_target_data, save_path, df, encode_series_mean,
                                   encode_series_std)

