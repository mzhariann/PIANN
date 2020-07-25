from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, TimeDistributed, Average, Embedding,Flatten
import numpy as np
import os
from utils import *
import keras.backend as K
from utils import get_h52_middle_layer

class CombiModel44:

    def __init__(self, exog_array, args, h522_model,seed):
        self.exog_array = exog_array
        self.pred_steps = args.pred_steps
        days = 364# 365
        len = days - self.pred_steps - self.pred_steps
        len2 = days - self.pred_steps
        n1 = 512 # args.n1
        dropout_rate2 = 0.1 # args.dropout_rate2

        # extract the last 14 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        #model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/h522_delta16_adam_hypertune/h522_delta16_155560_17/bestmodel.hdf5'
        #model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/h522_delta8_adam_hypertune/h522_delta8_155542_17/bestmodel.hdf5'
        #model_path = '/ivi/ilps/personal/mariann/pakdd/favorita/h522_delta2_adam_hypertune/h522_delta2_155525_13/bestmodel.hdf5'

        model_path = 'dh_h522_delta2/bestmodel.hdf5'

        #model_path = "result_simple_earlystopping_H522_81099_2/bestmodel.hdf5"
        #model_path = "h522_16_res/result_simple_earlystopping_H522_16_91628_1/bestmodel.hdf5"
        #model_path = "/ivi/ilps/personal/mariann/ah/result_h522_764_128886_1/bestmodel.hdf5"
        h522_model.load_weights(model_path)

        #pred_seq_train2 = Lambda(slice, arguments={'seq_length': self.pred_steps}, name='wavenet_slice_last_8')(x)
        pred_seq_train2 = Input(shape=(self.pred_steps, 1),name='wave_out')
        exog_seq = Input(shape=(len2, self.exog_array[0].shape[2]),name='other_stores_input')

        item_str = Input(shape=(1,),name='item_id_input')

        emb_1 = h522_model.get_layer('embedding_1')
        emb_1.name = 'item_embedding'
        emb_1.trainable = False
        flt_1 = h522_model.get_layer('flatten_1')
        flt_1.name = 'flatten_item_embedding'
        lmb1 = h522_model.get_layer('lambda_1')
        lmb1.name = 'tile_item_embedding'
        lmb2 = h522_model.get_layer('lambda_2')
        lmb2.name = 'reshape_item_embedding'
        con1 = h522_model.get_layer('concatenate_1')
        con1.name = 'concat_other_stores_item_embedding'
        td1 = h522_model.get_layer('time_distributed_1')
        td1.name = 'exog_data_td_128'
        td1.trainable = False
        dr1 = h522_model.get_layer('dropout_1')
        dr1.seed = seed
        dr1.name = 'exog_data_dropout'
        td2 = h522_model.get_layer('time_distributed_2')
        td2.name = 'exog_data_td_1'
        td2.trainable = False

        x4 = exog_seq
        x2 = emb_1(item_str)
        x2 = flt_1(x2)
        x3 = lmb1(x2)
        x3 = lmb2(x3)
        x4 = con1([x4, x3])

        x4 = td1(x4)
        x4 = dr1(x4)
        x4 = td2(x4)

        pred_seq_train = Lambda(slice, arguments={'seq_length': self.pred_steps}, name='ffe_slice_last_8')(x4)

        x4 = Add(name='add_wavenet_ffe')([pred_seq_train, pred_seq_train2])
        x4 = TimeDistributed(Dense(n1, activation='relu'),name = 'final_td_128')(x4)
        x4 = Dropout(dropout_rate2,name ='final_dropout',seed=seed)(x4)
        x4 = TimeDistributed(Dense(1),name='final_td_1')(x4)

        final_out = Lambda(slice, arguments={'seq_length': self.pred_steps},name='slice_last_8')(x4)
        self.model = Model([pred_seq_train2, exog_seq, item_str], final_out)


    def get_train_data(self, series_array, date_to_index, train_enc_start, train_enc_end, train_pred_start,
                       train_pred_end):
        exog_inds = date_to_index[train_enc_start:train_pred_end]

        encoder_input_data = get_time_block_series(series_array, date_to_index,
                                                   train_enc_start, train_enc_end)
        encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
        decoder_target_data = get_time_block_series(series_array, date_to_index,
                                                    train_pred_start, train_pred_end)
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)
        encoder_exog_data = self.exog_array[0][:, exog_inds, :]

        return [encoder_input_data,encoder_exog_data,self.exog_array[1]], decoder_target_data, encode_series_mean, encode_series_std


    def get_test_data(self, series_array, date_to_index, val_enc_start, val_enc_end, val_pred_start, val_pred_end):

        exog_inds = date_to_index[val_enc_start:val_pred_end]

        encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)
        encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
        decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)
        encoder_exog_data = self.exog_array[0][:, exog_inds, :]

        return [encoder_input_data,encoder_exog_data,self.exog_array[1]], decoder_target_data, encode_series_mean, encode_series_std

    def predict_sequence(self, input_sequence, encode_exog, encode_item):
        history_sequence = input_sequence.copy()
        pred_sequence = self.model.predict([history_sequence, encode_exog, encode_item])[:, -self.pred_steps:]
        return pred_sequence


    def predict_and_write(self, encoder_input_data, decoder_target_data, sample_ind, path, df, encode_series_mean,
                          encode_series_std):
        encode_series = encoder_input_data[0][sample_ind:sample_ind + 1, :, :]
        encode_exog = encoder_input_data[1][sample_ind:sample_ind + 1, :, :]
        pred_series = self.predict_sequence(encode_series,encode_exog, encoder_input_data[2][sample_ind:sample_ind + 1, :])

        #print(pred_series.shape)
        pred_series = (pred_series.reshape(-1, 1) * encode_series_std[sample_ind]) + encode_series_mean[sample_ind]
        target_series = (decoder_target_data[sample_ind, :, :1].reshape(-1, 1) * encode_series_std[sample_ind]) + \
                        encode_series_mean[sample_ind]

        #delo = df.loc[sample_ind]['item_delo'].split('_')[0]
        #item = df.loc[sample_ind]['item_delo'].split('_')[1].split('.')[0]
        delo = df.loc[sample_ind]['item_str'].split('_')[0]
        item = df.loc[sample_ind]['item_str'].split('_')[1]

        wdf = pd.DataFrame({'p': pred_series.flatten(), 'real': target_series.flatten()})
        if not os.path.exists(path + str(delo)):
            os.makedirs(path + str(delo))
        wdf.to_csv(path + str(delo) + '/' + str(item) + '.csv', index=False)

    def predict_all(self,encoder_input_data, decoder_target_data, path, df, encode_series_mean,
                                   encode_series_std):
        encode_series = encoder_input_data[0]#[sample_ind:sample_ind + 1, :, :]
        encode_exog = encoder_input_data[1]#[sample_ind:sample_ind + 1, :, :]
        pred_series_all = self.predict_sequence(encode_series, encode_exog,
                                            encoder_input_data[2])#[sample_ind:sample_ind + 1, :])

        # print(pred_series.shape)
        for sample_ind in range(df.shape[0]):
            if sample_ind%10 ==0:
                print(sample_ind)
            pred_series = pred_series_all[sample_ind:sample_ind+1,:]
            pred_series = (pred_series.reshape(-1, 1) * encode_series_std[sample_ind]) + encode_series_mean[sample_ind]
            target_series = (decoder_target_data[sample_ind, :, :1].reshape(-1, 1) * encode_series_std[sample_ind]) + \
                            encode_series_mean[sample_ind]

            # delo = df.loc[sample_ind]['item_delo'].split('_')[0]
            # item = df.loc[sample_ind]['item_delo'].split('_')[1].split('.')[0]
            delo = df.loc[sample_ind]['item_str'].split('_')[0]
            item = df.loc[sample_ind]['item_str'].split('_')[1]

            wdf = pd.DataFrame({'p': pred_series.flatten(), 'real': target_series.flatten()})
            if not os.path.exists(path + str(delo)):
                os.makedirs(path + str(delo))
            wdf.to_csv(path + str(delo) + '/' + str(item) + '.csv', index=False)
