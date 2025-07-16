import argparse
import yaml
import subprocess

# 读取YAML配置文件
def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="读取配置文件")
    parser.add_argument('--config',default='config_ty.yaml', type=str, help='YAML配置文件路径')
    args = parser.parse_args()
    return args

import pandas as pd


import subprocess
import os
import shutil

def run_scripts(site_name, start_date, end_date):
    # # 定义虚拟环境路径
    # VENV_PATH = "/home/ubuntu/accurain_predict/env_readdata"

    # # 激活虚拟环境
    # print("激活虚拟环境...")
    # activate_env = f"source {VENV_PATH}/bin/activate"
    # subprocess.run(activate_env, shell=True, check=True)

    # 运行 获取数据 脚本
    print("运行 获取数据 脚本...")
    get_db_file = f"./tmp/{site_name}.csv"
    if os.path.exists(get_db_file):
        os.remove(get_db_file)
    
    subprocess.run(f"python get_data.py --name {site_name} --start {start_date} --end {end_date}", shell=True, check=True)

    # 运行 分割数据 脚本
    print("运行 分割数据 脚本...")
    site_tmp_dir = f"./tmp/{site_name}"
    if os.path.exists(site_tmp_dir):
        shutil.rmtree(site_tmp_dir)

    subprocess.run(f"python pred_data_process.py --name {site_name} --file {get_db_file} --main split --start_time {start_date} --end_time {end_date}", shell=True, check=True)

    # 运行 预测模型 脚本
    print("运行 预测模型 脚本...")
    result_dir = "./results/informer_JFNG_data_15min_unwind_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"
    # 删除旧的预测结果文件
    for file in os.listdir(result_dir):
        if file.startswith(f"real_prediction_{site_name}_"):
            os.remove(os.path.join(result_dir, file))

    subprocess.run(f"python test02_copy.py --file_path ./tmp/{site_name} --seq_len 24 --label_len 12 --pred_len 4 --d_model 512 --n_heads 8 --e_layers 3 --d_layers 3 --d_ff 2048 --distil True --mix True --des 3", shell=True, check=True)

    # 运行 结果处理 脚本
    print("运行 结果处理 脚本...")
    subprocess.run(f"python pred_data_process.py --name {site_name} --file 'None' --main merge --start_time {start_date} --end_time {end_date}", shell=True, check=True)

    print("结果处理完成")
    print("脚本执行完成")



def read_rainfall_data(file_path):
    """
    读取包含时间和降雨值的数据文件（CSV 或 XLSX）。
    
    参数:
    file_path (str): 文件的路径
    
    返回:
    pandas.DataFrame: 包含时间和降雨值的 DataFrame
    """
    # 判断文件类型
    if file_path.endswith('.csv'):
        # 读取CSV文件
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        # 读取XLSX文件
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and XLSX are supported.")
    
    # 确保数据有两列：时间和降雨值
    if data.shape[1] != 2:
        raise ValueError("The file must have exactly two columns: time and rainfall.")
    
    # 给列命名（如果没有命名列的话）
    data.columns = ['time', 'rainfall']
    
    # 转换时间列为日期时间格式（如果尚未转换）
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    
    # 处理降雨值为数值类型（如果有缺失值或非数字的情况）
    data['rainfall'] = pd.to_numeric(data['rainfall'], errors='coerce')
    # 补充缺失值或者Nan，补充为0
    data['rainfall'].fillna(0, inplace=True)

    # 返回处理后的数据
    return data

    

# 主程序
def main():
    # 获取命令行参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 获取配置项
    site_names = config['site_names']
    reference_data_path = config['reference_data_path']
    site_coordinates = config['site_coordinates']
    reference_site_coordinates = config['reference_site_coordinates']
    predata_type =  config['predata_type']
    start_date = config['start_date']   
    end_date = config['end_date']


    # 打印获取的参数
    print("站点名列表:", site_names)
    print("参考数据文件夹路径:", reference_data_path)
    print("站点经纬度列表:", site_coordinates)
    print("参考站点经纬度列表:", reference_site_coordinates)
    print("预测类型:", predata_type)
    
    # 运行bash脚本
    for site in site_names:
        run_scripts(site,start_date,end_date)

    #  读取预测数据
    pred_data = []
    for site in site_names:
        pred_data.append(read_rainfall_data(f"./pred_output/{site}/{site}_pred{predata_type}.csv"))



if __name__ == '__main__':
    main()
