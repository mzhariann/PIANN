from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, TimeDistributed, Average, Embedding,Flatten
import numpy as np
import os
from utils import *
import keras.backend as K
from utils import get_h52_middle_layer

class CombiModel82:

    def __init__(self,args,seed,teacher):
        self.pred_steps = args.pred_steps
        self.teacher = teacher
        # extract the last 14 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        dropout_rate = 0.1#args.dropout_rate

        history_seq = Input(shape=(self.pred_steps, 1),name='history_input')
        x = history_seq

        x4_128 = TimeDistributed(Dense(args.n1, activation='relu'), name='first_td_128')(x)
        x4 = Dropout(dropout_rate, name='first_dropout', seed=seed)(x4_128)
        x4_128 = TimeDistributed(Dense(args.n2, activation='relu'), name='second_td_128')(x4)
        x4 = Dropout(dropout_rate, name='second_dropout', seed=seed)(x4_128)
        x4_128 = TimeDistributed(Dense(args.n3, activation='relu'), name='final_td_128')(x4)
        x4 = Dropout(dropout_rate, name='final_dropout', seed=seed)(x4_128)
        x4_1 = TimeDistributed(Dense(1), name='final_td_1')(x4)

        final_out = Lambda(slice, arguments={'seq_length': self.pred_steps},name='slice_last_8')(x4_1)
        self.model = Model(history_seq, final_out)


    def get_train_data(self, series_array, date_to_index, train_enc_start, train_enc_end, train_pred_start,
                       train_pred_end):

        encoder_input_data = get_time_block_series(series_array, date_to_index,
                                                   train_enc_start, train_enc_end)
        encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
        decoder_target_data = get_time_block_series(series_array, date_to_index,
                                                    train_pred_start, train_pred_end)
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)

        return encoder_input_data, decoder_target_data, encode_series_mean, encode_series_std


    def get_test_data(self, series_array, date_to_index, val_enc_start, val_enc_end, val_pred_start, val_pred_end):

        encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)
        encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
        decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)

        return encoder_input_data, decoder_target_data, encode_series_mean, encode_series_std

    def predict_sequence(self, input_sequence):
        history_sequence = input_sequence.copy()
        pred_sequence = self.model.predict(history_sequence)[:, -self.pred_steps:]

        ftd_128= self.teacher.get_layer('final_td_128')
        fdo = self.teacher.get_layer('final_dropout')
        ftd_1 = self.teacher.get_layer('final_td_1')

        x = ftd_128(K.variable(pred_sequence))
        x = fdo(x)
        x = ftd_1(x)
        sess = K.get_session()
        with sess.as_default():
            t = x.eval()
        pred_sequence = t[:, -self.pred_steps:]
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
        encode_series = encoder_input_data
        pred_series_all = self.predict_sequence(encode_series)#[sample_ind:sample_ind + 1, :])

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
