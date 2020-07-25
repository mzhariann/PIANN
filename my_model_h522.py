from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, TimeDistributed, Average, Embedding,Flatten
import numpy as np
import os
from utils import *
import keras.backend as K

class MyModelH522:

    def __init__(self, exog_array, args):
        self.exog_array = exog_array
        self.pred_steps = args.pred_steps
        self.embed_size = args.embed_size
        days = 365#1034#365
        len = days - self.pred_steps
        dropout_rate = args.dropout_rate
        dense_layer1_nuerons = args.dense_1

        history_seq = Input(shape=(None, 1))

        exog_seq = Input(shape=(len, self.exog_array[0].shape[2]))
        x4 = exog_seq

        item_str = Input(shape=(1,))
        #x2 = Embedding(143, 4, input_length=1)(item_str)
        x2 = Embedding(1656, self.embed_size, input_length=1)(item_str)
        #x2 = Embedding(50, self.embed_size, input_length=1)(item_str)
        #x2 = Embedding(81, self.embed_size, input_length=1)(item_str)
        #x2 = Embedding(114, self.embed_size, input_length=1)(item_str)
        #x2 = Embedding(1115, 6, input_length=1)(item_str)
        x2 = Flatten()(x2)
        x3 = Lambda(K.tile, arguments={'n': (1, len)})(x2)
        x3 = Lambda(K.reshape, arguments={'shape': (-1,len, self.embed_size)})(x3)

        x4 = Concatenate()([x4, x3])

        x4 = TimeDistributed(Dense(dense_layer1_nuerons, activation='relu'))(x4)
        x4 = Dropout(dropout_rate)(x4)
        x4 = TimeDistributed(Dense(1))(x4)

        self.model = Model([history_seq, exog_seq,item_str], x4)

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
        pred_sequence = self.model.predict([history_sequence,encode_exog, encode_item])[:,-self.pred_steps:]
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
