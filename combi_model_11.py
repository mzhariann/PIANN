from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, TimeDistributed, Average, Embedding,Flatten
import os
from utils import *

class CombiModel11:

    def __init__(self, args,seed):
        self.pred_steps = args.pred_steps
        days = 365#1034#365
        len = days - self.pred_steps - self.pred_steps

        n_filters = args.n_filters  # 32
        filter_width = args.filter_width  # 2
        n_dilation = args.n_dilation
        dilation_rates = [2 ** i for i in range(n_dilation)]  # [2 ** i for i in range(6)]
        dense_layer1_nuerons = args.dense_1
        dropout_rate = args.dropout_rate

        history_seq = Input(shape = (len, 1))
        #history_seq = Input(shape=(333, 1))
        #history_seq = Input(shape=(748, 1))
        x = history_seq

        conv_layers = []
        for dilation_rate in dilation_rates:
            conv_layers.append(Conv1D(filters=n_filters,
                                      kernel_size=filter_width,
                                      padding='causal',
                                      dilation_rate=dilation_rate))

        dense_layer1 = TimeDistributed(Dense(dense_layer1_nuerons, activation='relu'))
        dropout_layer = Dropout(dropout_rate,seed=seed)
        dense_layer2 = TimeDistributed(Dense(1))

        # extract the last 14 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        lambda_layer = Lambda(slice, arguments={'seq_length': 1})
        lambda_layer2 = Lambda(slice, arguments={'seq_length': len})
        #lambda_layer2 = Lambda(slice, arguments={'seq_length': 333})
        #lambda_layer2 = Lambda(slice, arguments={'seq_length': 748})
        concat_layer = Concatenate(axis=1)

        for p in range(self.pred_steps):
            x2 = x
            for c in conv_layers:
                x2 = c(x2)
            x2 = dense_layer1(x2)
            x2 = dropout_layer(x2)
            x2 = dense_layer2(x2)

            sliced_x2 = lambda_layer(x2)
            x = concat_layer([x, sliced_x2])
            x = lambda_layer2(x)

        pred_seq_train = Lambda(slice, arguments={'seq_length': self.pred_steps})(x)
        self.model = Model(history_seq, pred_seq_train)

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
        pred_sequence = self.model.predict(history_sequence)[:,-self.pred_steps:]
        return pred_sequence


    def predict_and_write(self, encoder_input_data, decoder_target_data, sample_ind, path, df, encode_series_mean,
                          encode_series_std):
        encode_series = encoder_input_data[sample_ind:sample_ind + 1, :, :]
        pred_series = self.predict_sequence(encode_series)

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
        pred_series_all = self.predict_sequence(encode_series)

        #df2 = pd.read_csv('data/ah_online_sales_143_764.csv')
        # print(pred_series.shape)
        for sample_ind in range(df.shape[0]):
            if sample_ind%10 ==0:
                print(sample_ind)
            pred_series = pred_series_all[sample_ind:sample_ind+1,:]
            pred_series = (pred_series.reshape(-1, 1) * encode_series_std[sample_ind]) + encode_series_mean[sample_ind]
            target_series = (decoder_target_data[sample_ind, :, :1].reshape(-1, 1) * encode_series_std[sample_ind]) + \
                            encode_series_mean[sample_ind]

            #delo = df.loc[sample_ind]['item_delo'].split('_')[0]
            #item = df.loc[sample_ind]['item_delo'].split('_')[1].split('.')[0]
            delo = df.loc[sample_ind]['item_str'].split('_')[0]
            item = df.loc[sample_ind]['item_str'].split('_')[1]
            #if df.loc[sample_ind]['item_str'] in df2['item_str'].tolist():
            wdf = pd.DataFrame({'p': pred_series.flatten(), 'real': target_series.flatten()})
            if not os.path.exists(path + str(delo)):
                os.makedirs(path + str(delo))
            wdf.to_csv(path + str(delo) + '/' + str(item) + '.csv', index=False)
