import pandas as pd
import numpy as np
import tqdm
import pathlib
import argparse
import re
import shutil

def preprocess_data(file):
    df = pd.read_csv(file)
    # df = df.drop(columns='wind_speed')
    df  = df[(df['date'] != 'date')]
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates()
    return df

def split_csv(file, site_name):
    if not pathlib.Path(file).exists():
        print(f"File {file} does not exist")
        exit()

    # 补全时间
    # df.set_index('date', inplace=True)
    # full_range = pd.date_range(start=start_time, end=end_time, freq='15min')
    # full_data = df.reindex(full_range)

    # full_data.reset_index(inplace=True)
    # full_data.rename(columns={'index': 'date'}, inplace=True)


    # Load the csv file as a pandas DataFrame
    data = preprocess_data(file)


    # Get the total number of rows in the csv file
    total_rows = len(data)

    window_size = 1

    # Set the batch size
    batch_size = 60

    # Calculate the number of batches
    num_batches = total_rows // window_size

    out_dir = pathlib.Path(f'./tmp/{site_name}/')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each batch
    for i in tqdm.tqdm(range(num_batches), desc="Processing batches"):
        # Get the start and end indices for the current batch
        start_index = i * window_size
        end_index = (i * window_size) + batch_size

        # Extract the data for the current batch
        batch_data = data.iloc[start_index:end_index]
        if len(batch_data) != batch_size:
            # print(f"Skipping batch {i+1} as it is smaller than the batch size")
            # print(f"Batch size: {batch_size}, Batch data size: {len(batch_data)}")
            continue
        # Generate a new csv file for the current batch

        batch_data.to_csv(pathlib.Path(out_dir,f'{site_name}_{i+1}.csv'), index=False, encoding='utf-8')
        # print(start_index,end_index)
        # print(batch_data)

    print(pathlib.Path.cwd().joinpath(out_dir))

def custom_sort_key(filename):
    # 使用正则表达式提取文件名中的数字部分
    parts = re.findall(r'(\D+)(\d+)', filename)
    # 将每个部分转换为元组，其中数字部分转换为整数
    return tuple((part, int(num)) for part, num in parts)

def list_test_files(directory, site_name):
    # 创建 Path 对象
    dir_path = pathlib.Path(directory)

    # 确保目录存在
    if not dir_path.is_dir():
        raise ValueError(f"指定的路径不是一个目录: {directory}")

    # 使用 glob 方法获取所有以 'test' 开头的文件
    test_files = sorted(dir_path.glob(f'real_prediction_{site_name}_*'), key=lambda x: custom_sort_key(x.name))

    return test_files

def merge_csv(site_name, start_time, end_time):

    folder_path = list_test_files(f"./results/informer_{site_name}_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_a_0", site_name)

    pred_h = {"15min":1, "1h":4, "2h":8, "3h":12}

    key_list = list(pred_h.keys())
    for key in key_list:
        pred_hour = pred_h[key]

        data = pd.DataFrame()

        a = 0
        for npy_file in tqdm.tqdm(folder_path, desc="npy files"):
            if a % pred_hour +1 != 1:
                a += 1
                continue
            npy_data = np.load(npy_file)
            a += 1
            # print(npy_file)

            npy_data = np.reshape(npy_data, (12, 1))
            # npy_data = np.reshape(npy_data, (4, 1))

            npy_df = pd.DataFrame(npy_data)

            npy_df = npy_df.iloc[0:pred_hour]

            # data = data.append(npy_df, ignore_index=True)
            data = pd.concat([data, npy_df], ignore_index=True)
        data.columns = ['pred']
        data['pred'] = data['pred'].astype(float)
        data = data.reset_index(drop=True)
        print(len(data['pred']))
        print(data.head())

        file = f'./tmp/{site_name}.csv'
        if not pathlib.Path(file).exists():
            print(f"File {file} does not exist")
            exit()

        df = pd.read_csv(file)

        df = df[['date','tp']]
        df  = df[(df['date'] != 'date')]
        df['date'] = pd.to_datetime(df['date'])
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        df = df.drop_duplicates()

        # TODO: 如果前面分片时裁剪了时间，这里也需要裁剪，反之注释掉即可
        # df = df[df['date'] <= end_time]
        # df = df[df['date'] >= start_time]

        df = df.iloc[60:len(data)+60]
        df = df.reset_index(drop=True)
        # print(len(df))
        # print(df.head())

        data_new = pd.merge(data, df, right_index=True, left_index=True, how='outer')
        # print(data_new.head())
        
        data_new = data_new[data_new['date'] >= start_time]
        data_new = data_new[data_new['date'] <= end_time]
        data_new['pred'] = data_new['pred'].astype(float)

        data_new = data_new[['date','tp' ,'pred']]
        data_new.rename(columns={'tp':'true'}, inplace=True)
        data_new['pred'] = data_new['pred'].astype(float).round(4)
        data_new['true'] = data_new['true'].astype(float).round(4)
        # data_new['pred'] = data_new['pred'].clip(lower=0.001)
        # data_new['true'] = data_new['true'].clip(lower=0.001)
        data_new['pred'] = data_new['pred'].mask(data_new['pred'] < 0.01, 0)
        data_new['true'] = data_new['true'].mask(data_new['true'] < 0.01, 0)
        # data_new = data_new[data_new['date'] >= start_time]

        out_dir = pathlib.Path(f'./pred_output/{site_name}')
        if not out_dir.exists():
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        data_new.to_csv(pathlib.Path(out_dir, f'{site_name}_pred{pred_hour}by{pred_hour}.csv'), index=False, encoding='utf-8')
        # print(pathlib.Path.cwd().joinpath(out_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据导出')
    parser.add_argument('--name', type=str, required=True, help='站点名称：如B19')
    parser.add_argument('--file', type=str, required=True, help='csv文件路径')
    parser.add_argument('--main', type=str, required=True, choices=["split","merge"], help='入口函数')
    parser.add_argument('--start_time', type=str, required=True, help='开始时间,如2024-04-18')
    parser.add_argument('--end_time', type=str, required=True, help='结束时间,如2024-09-29')

    global start_date,end_date
    args = parser.parse_args()

    file = args.file
    site_name = args.name
    start_date = args.start_time
    end_date = args.end_time
    if args.main == 'split':
        split_csv(file, site_name)
    else:
        merge_csv(site_name, start_date, end_date)