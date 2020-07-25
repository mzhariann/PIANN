from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.models import Model


def get_item_ohe_exog_array(df_input):
    df = df_input.copy()
    df['item_delo'] = df['item_delo'].apply(lambda x: x.split('_')[1])
    item_ohe = pd.get_dummies(df['item_delo'])
    exog_array = np.empty((df.shape[0], df.shape[1]-1,item_ohe.shape[1]))
    for i in range(df.shape[1]-1):
        exog_array[:,i,:] = item_ohe.values
    return exog_array

def get_dayofweek_exog_array(df):
    dow_ohe = pd.get_dummies(pd.to_datetime(df.columns[1:]).dayofweek)
    dow_array = np.expand_dims(dow_ohe.values, axis=0)
    exog_array = np.tile(dow_array, (df.shape[0], 1, 1))
    return exog_array

def get_exog_array_from_xgb_features(df, item_features_path, features):

    exog_array = np.empty((df.shape[0], df.shape[1] - 1, len(features)))
    for i in range(df.shape[0]):
        delo = df.loc[i]['item_delo'].split('_')[0]
        item = df.loc[i]['item_delo'].split('_')[1].split('.')[0]
        tmp_features_df = pd.read_csv(item_features_path + str(int(delo)) + '/' + str(item) + '.csv')
        tmp_features_df = tmp_features_df[features]
        exog_array[i, :, :] = tmp_features_df.values
    return exog_array

def get_exog_array_other_stores(df, item_features_path, num_stores=49):

    exog_array = np.empty((df.shape[0], df.shape[1] - 1, num_stores))
    for i in range(df.shape[0]):
        tmp_features_df = pd.read_csv(item_features_path + df.loc[i]['item_str'] + '.csv')
        exog_array[i, :, :] = tmp_features_df.values.T
    return exog_array

def get_exog_array_other_stores_with_item_str(df2, item_features_path, num_stores=49):

    df = df2.copy()

    exog_array = np.empty((df.shape[0], df.shape[1] - 1, num_stores))
    item_list = np.empty((df.shape[0] ,1))
    store_list = np.empty((df.shape[0], 1))

    f = lambda x: x.split('_')[0]
    f2 = lambda x: x.split('_')[1]
    df['store'] = df['item_str'].apply(f)
    df['item'] = df['item_str'].apply(f2)
    df['item_cat'] = df['item'].astype('category')
    df['store_cat'] = df['store'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)



    for i in range(df.shape[0]):
        tmp_features_df = pd.read_csv(item_features_path + df.loc[i]['item_str'] + '.csv')
        delo = df.loc[i]['store_cat']
        item = df.loc[i]['item_cat']
        #print(tmp_features_df.shape)
        exog_array[i, :, :] = tmp_features_df.values.T
        item_list[i] = int(item)
        store_list[i] = int(delo)
    return [exog_array,item_list,store_list]

def get_dates(df,pred_steps):

    data_start_date = df.columns[1]
    data_end_date = df.columns[-1]

    pred_length = timedelta(pred_steps)

    first_day = pd.to_datetime(data_start_date)
    last_day = pd.to_datetime(data_end_date)

    val_pred_start = last_day - pred_length + timedelta(1)
    val_pred_end = last_day

    train_pred_start = val_pred_start - pred_length
    train_pred_end = val_pred_start - timedelta(days=1)

    enc_length = train_pred_start - first_day

    train_enc_start = first_day
    train_enc_end = train_enc_start + enc_length - timedelta(1)

    val_enc_start = train_enc_start + pred_length
    val_enc_end = val_enc_start + enc_length - timedelta(1)

    return train_enc_start, train_enc_end, train_pred_start, train_pred_end, val_enc_start, val_enc_end, \
           val_pred_start, val_pred_end


def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]


def transform_series_encode(series_array):
    #series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1, 1)
    series_std = series_array.std(axis=1).reshape(-1, 1)
    #print(series_std.shape)
    print(np.where(series_std == 0))
    series_array = (series_array - series_mean) / series_std
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array, series_mean, series_std


def transform_series_decode(series_array, encode_series_mean, encode_series_std):
    #series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = (series_array - encode_series_mean) / encode_series_std
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array


def plot_history(save_path,history,y1,y2):
    if y2 == 'val_loss':
        file_path = save_path+'val_loss.txt'
        with open(file_path,"w") as log_file:
            for l in history.history['val_loss']:
                log_file.write(str(l) + ",")
            log_file.write("\n")
    plt.clf()
    plt.plot(history.history[y1])
    plt.plot(history.history[y2])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Loss Over Time')
    plt.legend([y1, y2])
    plt.savefig(save_path+y1+'_'+y2+'_history.png', dpi=100)


def get_h52_results(encoder_input_data, model,pred_steps):
    #model_path = "result_simple_earlystopping_H52_81930_1/bestmodel.hdf5"
    model_path = "result_simple_earlystopping_H522_81099_2/bestmodel.hdf5"
    model.load_weights(model_path)
    pred_sequence = model.predict(encoder_input_data)[:, -pred_steps:]
    return pred_sequence

def get_h52_results_oi(encoder_input_data, model,pred_steps):
    #model_path = "result_simple_earlystopping_H52_81930_1/bestmodel.hdf5"
    model_path = "result_simple_earlystopping_H522_items_88060_1/bestmodel.hdf5"
    model.load_weights(model_path)
    pred_sequence = model.predict(encoder_input_data)[:, -pred_steps:]
    return pred_sequence

def get_h52_middle_layer(encoder_input_data, model,pred_steps):
    model_path = "result_simple_earlystopping_H522_81099_2/bestmodel.hdf5"
    model.load_weights(model_path)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('time_distributed_2').output)
    intermediate_output = intermediate_layer_model.predict(encoder_input_data,steps=32)#[:, -pred_steps:]
    return intermediate_output


def get_h52_embeddings(encoder_input_data, model,pred_steps):
    #model_path = "result_simple_earlystopping_H52_81930_1/bestmodel.hdf5"
    model_path = "result_simple_earlystopping_H522_81099_2/bestmodel.hdf5"
    model.load_weights(model_path)
    embeddings = model.get_layer('embedding_1')
    return embeddings

def get_simple_results(encoder_input_data, model,pred_steps):
    #model_path = "result_simple_earlystopping_H52_81930_1/bestmodel.hdf5"
    model_path = "result_simple_earlystopping_simple_notf_76494_1/bestmodel.hdf5"
    model.load_weights(model_path)
    pred_sequence = model.predict(encoder_input_data)
    return pred_sequence

def get_simple_middle_layer(encoder_input_data, model,pred_steps):
    model_path = "result_simple_earlystopping_simple_notf_76494_1/bestmodel.hdf5"
    model.load_weights(model_path)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('dense_1').get_output_at(-1))
    intermediate_output = intermediate_layer_model.predict(encoder_input_data)[:, -pred_steps:]
    return intermediate_output

def parse_args():
    parser = argparse.ArgumentParser(description='Mozhdeh')
    parser.add_argument('-job_id', type=str, default="")
    parser.add_argument('-pred_steps', type=int, default=8)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-n_filters', type=int, default=32)
    parser.add_argument('-filter_width', type=int, default=2)
    parser.add_argument('-n_dilation', type=int, default=6)
    parser.add_argument('-dense_1', type=int, default=128)
    parser.add_argument('-dropout_rate', type=float, default=0.2)
    parser.add_argument('-dropout_rate2', type=float, default=0.2)

    parser.add_argument('-n_filters2', type=int, default=32)
    parser.add_argument('-filter_width2', type=int, default=2)
    parser.add_argument('-n_dilation2', type=int, default=6)

    parser.add_argument('-w1', type=float, default=0.5)
    parser.add_argument('-w2', type=float, default=0.5)
    parser.add_argument('-w3', type=float, default=0.5)

    parser.add_argument('-n1', type=int, default=128)
    parser.add_argument('-n2', type=int, default=128)
    parser.add_argument('-n3', type=int, default=128)

    parser.add_argument('-embed_size',type=int,default=7)

    args = parser.parse_args()
    return args
