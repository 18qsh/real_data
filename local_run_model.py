import pandas as pd
import numpy as np
import time
import csv
import datetime
import os
import subprocess
import json
import calendar
import time
from math import *
import threading
import queue
import sys
import clickhouse_connect
import runpy
import mysql.connector


# 分割时间
def split_time(start_timestamp, end_timestamp):
    # 默认分为八段
    ranges_list = np.linspace(start_timestamp, end_timestamp, 8)
    return ranges_list


# 请求数据
def req_worker(start_t, end_t, device, result_queue, table_name):
    # 构建查询
    start_d = pd.to_datetime(start_t, unit='s', utc=True)
    end_d = pd.to_datetime(end_t, unit='s', utc=True)
    query = f"""
    SELECT *
    FROM accurain.dwd_{table_name}_data
    WHERE data_time >= toDateTime64('{start_d}') AND data_time <= toDateTime64('{end_d}') AND device = '{device}'
    ORDER BY data_time ASC
    """
    # ClickHouse连接信息 - 需要填写
    client = clickhouse_connect.get_client(host=clickhouse_host, port=clickhouse_port, user=clickhouse_user,
                                           password=clickhouse_password)
    try:
        # 执行查询
        result = client.query(query)
        data = result.result_rows
        column_names = [col.lower() for col in result.column_names]
        df = pd.DataFrame(data, columns=column_names)
        # data = connection.query(query)

        # 将结果加入队列
        result_queue.put(df)
        # 单线程

    except Exception as e:
        print(f"查询异常: {str(e)}")


# 合并结果
def req_queue(rangelist, device, table_name):
    # 定义线程数量
    num_threads = len(rangelist) - 1
    # 创建线程池
    threads = []
    result_queues = []
    for i in range(num_threads):
        q = queue.Queue()
        t = threading.Thread(target=req_worker, args=(rangelist[i], rangelist[i + 1], device, q, table_name))
        threads.append(t)
        result_queues.append(q)

    # 启动线程
    for t in threads:
        t.start()

    # 等待所有线程结束
    for t in threads:
        t.join()

    # 合并结果
    results = []
    for q in result_queues:
        while not q.empty():
            results.append(q.get())

    # 返回结果
    return pd.concat(results)


def read_database(table_name, start_t, end_t, site_name):
    # table_name 数据表名: gnss ; hws
    # start_t,end_t: 开始和结束时间戳
    # site_name: 读取设备号
    ranges_list = split_time(start_t, end_t)
    # 输出结果
    # for i in range(len(ranges_list) - 1):
    #     print(f"Range {i + 1}: ({ranges_list[i]}, {ranges_list[i + 1]})")
    # 记录开始时间
    b_time = time.time()
    # 多线程读取
    data = req_queue(ranges_list, site_name, table_name)
    # 单线程测试用
    # result_queues = []
    # data = req_worker(start_t='2025-03-01',end_t='2025-03-02',device='b13',result_queue=result_queues, table_name=f'accurain.dwd_{table_name}_data')

    # 记录结束时间
    e_time = time.time()
    print('获取表数据用时：' + str(e_time - b_time))
    return data


def dynamic_window(time_step, winsize, win_max, gnss_data, hws_data, resample_time):
    flag = 1
    while flag:
        t1 = resample_time - winsize  # i*5*60 5min interval；  【-0.5*60，0.5*60】window length
        t2 = resample_time + winsize
        tmp_data = gnss_data[(gnss_data['data_time'] > int(t1)) & (gnss_data['data_time'] < int(t2))]
        mean_data = tmp_data.loc[:, ['ztd', 'latitude', 'longitude', 'height']].mean(0)
        tmp_hwsdata = hws_data[(hws_data['data_time'] > int(t1)) & (hws_data['data_time'] < int(t2))]
        mean_hwsdata = (tmp_hwsdata.loc[:, ['ta', 'pa', 'ua', 'sm']].mean(0))
        winsize = winsize * 2
        if (tmp_data.shape[0]) >= 1 & tmp_hwsdata.shape[0] >= 1:
            flag = 0
        if winsize > win_max:
            flag = 0
    resample_time_utc = pd.to_datetime(resample_time, unit='s')
    t1 = resample_time - time_step / 2 * 60  # i*time_step*60 time_step minutes interval
    t2 = resample_time + time_step / 2 * 60
    tmp_hwsdata = hws_data[(hws_data['data_time'] > int(t1)) & (hws_data['data_time'] < int(t2))]
    Rc_data = tmp_hwsdata['rc'].values
    if Rc_data.size:
        Rc_diff = Rc_data[-1] - Rc_data[0]
        if Rc_diff < 0:  # 降雨传感器，每天零点会重置时间；因此跨越零点的时候，需要做特别处理
            max_v = np.max(Rc_data, axis=0)  # 找到前天的降雨积累最大值
            tmp_rf = max_v - Rc_data[0]  # 得到前一天的降雨差值
            Rc_diff = Rc_data[-1] + tmp_rf  # 得到最终的降雨差值
    else:
        Rc_diff = np.nan

    mean_hwsdata['rc'] = Rc_diff
    return mean_data, mean_hwsdata


def resampling(now_time, gnss_data, hws_data, time_step, winsize, win_max):
    # function: resample real_data and interpolate those missing values
    # input:
    # now_time, gnss_data, hws_data,
    # time_step: time interval for every data point (unit = minute)
    # winsize, win_max: averaging window size (unit = second)
    now_time_utc = pd.to_datetime(now_time, unit='s')
    near_minute = np.floor(now_time_utc.minute / time_step) * time_step
    end_time_utc = now_time_utc.replace(minute=near_minute.astype(int), second=0, microsecond=0)
    start_time_utc = end_time_utc - datetime.timedelta(minutes=seq_len * time_step)  # 根据seqlen=96，得到起始时间
    # resample_time = pd.date_range(start=start_time_utc, end=end_time_utc, freq='5min') # 产生重采样时间点集合
    end_time_unix = calendar.timegm(end_time_utc.timetuple())
    resample_time = calendar.timegm(start_time_utc.timetuple())
    resamp_data = pd.DataFrame(None,
                               index=['t2m', 'sp', 'rh', 'wind_speed', 'tp', 'ztd', 'latitude', 'longitude', 'height'])

    for i in range(seq_len):
        re_time = resample_time + (i) * time_step * 60
        # 动态改变窗口大小，获取时间窗口内平均值
        mean_data, mean_hwsdata = dynamic_window(time_step, winsize, win_max, gnss_data, hws_data, re_time)
        # 拼接数据
        s = pd.concat([mean_hwsdata, mean_data], axis=0).to_frame()
        s.index = ['t2m', 'sp', 'rh', 'wind_speed', 'tp', 'ztd', 'latitude', 'longitude', 'height']
        resamp_data = pd.concat([resamp_data, s], axis=1, ignore_index=True)

    resamp_data.loc[resamp_data.shape[0]] = np.arange(resample_time, end_time_unix, time_step * 60)
    resamp_data = resamp_data.rename(index={resamp_data.shape[0] - 1: 'date'})
    # resamp_data.drop(columns=0)
    resamp_data = resamp_data.T
    resamp_data['date'] = pd.to_datetime(resamp_data['date'], unit='s')
    return resamp_data


def calc_pwv(ztd, t, p, lat, height):
    # t: temperature (k)
    # p: pressure (hpa)
    # lat: latitude (degree)
    # height: (m)
    # ztd: zenith tropospheric delay (m)
    # Saastamoinen model
    tm = 70.2 + 0.72 * t
    lat = lat / 180 * pi
    zhd0 = pow(10, -3) * (2.2768 * p / (1 - 0.00266 * np.cos(2 * lat) - 0.00028 * height * pow(10, -3)))  # 单位m
    zwd0 = pow(10, 3) * (ztd - zhd0)  # %单位mm
    k = pow(10, 6) / (4.613 * pow(10, 6) * (3.776 * pow(10, 5) / tm + 22.1))  # 单位Mkg/m^3
    k = k * pow(10, 6) / pow(10, 3)  # 单位换算kg/m^2=mm
    pwv = k * zwd0
    return pwv


def get_data(now_time, history_time, site_name):
    gnss = 'gnss'
    hws = 'hws'
    end_t = now_time
    start_t = history_time
    gnss_data = read_database(gnss, start_t, end_t, site_name)
    hws_data = read_database(hws, start_t, end_t, site_name)
    if len(gnss_data) == 0 or len(hws_data) == 0:
        flag = -1
        return flag

    gnss_data['data_time'] = gnss_data['data_time'].astype(int) // 10 ** 9  # clickhouse的dateime64转为int时是纳秒级，除10的9次方变为秒级
    gnss_data['ztd'] = gnss_data['ztd'].astype(float)
    gnss_data['latitude'] = gnss_data['latitude'].astype(float)
    gnss_data['longitude'] = gnss_data['longitude'].astype(float)
    gnss_data['height'] = gnss_data['height'].astype(float)
    # gnss_data['time'] = pd.to_datetime(gnss_data['time'], unit='ms')
    # hws_data['time'] = pd.to_datetime(hws_data['time'], unit='ms')
    hws_data['data_time'] = hws_data['data_time'].astype('int64') // 10 ** 9
    hws_data['ta'] = hws_data['ta'].astype(float)
    hws_data['pa'] = hws_data['pa'].astype(float)
    hws_data['rc'] = hws_data['rc'].astype(float)
    hws_data['sa'] = hws_data['ua'].astype(float)
    hws_data['sm'] = hws_data['sm'].astype(float)
    # hws_data['time'] = pd.to_datetime(hws_data['time'], unit='s')
    # print(real_data)
    ##############
    resamp_data = resampling(now_time, gnss_data, hws_data, 15, 30, 30 * 2)  # 重采样数据
    """
    读取数据，写入csv文件中
    """
    resamp_data.iloc[:, 0:9] = resamp_data.iloc[:, 0:9].interpolate(method='linear', order=1, limit=10,
                                                                    limit_direction='both')  #
    resamp_data["t2m"] = resamp_data["t2m"].astype(float)
    resamp_data["t2m"] = resamp_data[["t2m"]].apply(lambda x: x["t2m"] + 273.15, axis=1)
    resamp_data["t2m"] = resamp_data[["t2m"]].apply(lambda x: round(x["t2m"], 2), axis=1)
    resamp_data["sp"] = resamp_data["sp"].astype(float)
    resamp_data["sp"] = resamp_data[["sp"]].apply(lambda x: round(x["sp"], 2), axis=1)
    resamp_data["rh"] = resamp_data["rh"].astype(float)
    resamp_data["rh"] = resamp_data[["rh"]].apply(lambda x: round(x["rh"], 2), axis=1)
    resamp_data["wind_speed"] = resamp_data["wind_speed"].astype(float)
    resamp_data["wind_speed"] = resamp_data[["wind_speed"]].apply(lambda x: round(x["wind_speed"], 2), axis=1)

    resamp_data["tp"] = resamp_data["tp"].astype(float)
    resamp_data["tp"] = resamp_data[["tp"]].apply(lambda x: round(x["tp"], 2), axis=1)
    resamp_data["tp"] = resamp_data[["tp"]].apply(lambda x: x["tp"] + 1e-5, axis=1)  # avoid nan when all zeros
    resamp_data.loc[10, "tp"] = resamp_data.loc[10, "tp"] + 1e-4
    # calculate PWV
    ztd_data = resamp_data.loc[:, 'ztd']
    t_data = resamp_data.loc[:, 't2m']
    p_data = resamp_data.loc[:, 'sp']
    lat_data = resamp_data.loc[:, 'latitude']
    h_data = resamp_data.loc[:, 'height']
    pwv = calc_pwv(ztd_data, t_data, p_data, lat_data, h_data)
    #
    data_csv = resamp_data[['date', 't2m', 'sp', 'rh']]  # 重组数据
    data_csv.insert(4, 'pwv', pwv)
    data_csv.loc[:, "pwv"] = data_csv.loc[:, "pwv"].astype(float)
    # data_csv["pwv"] = data_csv[["pwv"]].apply(lambda x: round(x["pwv"], 2), axis=1)
    data_csv.loc[:, "pwv"] = data_csv["pwv"].round(2)
    data_csv.insert(5, 'tp', resamp_data.loc[:, 'tp'])  # 重组数据
    data_csv.to_csv('./real_data/{}.csv'.format(site_name), index=False)
    if data_csv.isna().any().any():
        flag = 0
        print('data have null value,please check in real_data/{}.csv'.format(site_name))
    else:
        flag = 1
        print('read data done')
    return flag


def run_model(site_name):
    # 更改运行模型脚本方式，原来使用cmd不利于调试
    args = ['run_model.py', '--data_path', f'{site_name}.csv']
    original_argv = sys.argv.copy()
    sys.argv = args
    try:
        # 设置新的sys.argv
        sys.argv = args
        # 执行脚本，输出会直接显示在当前位置
        runpy.run_path('run_model.py', run_name='__main__')
        retval = 0  # 如果正常退出
    except SystemExit as e:
        # 捕获脚本中的sys.exit调用
        retval = e.code
    finally:
        # 恢复原来的sys.argv
        sys.argv = original_argv
    print(f"模型执行完毕，返回码: {retval}")
    # 通过subprocess.popen 执行 命令行命令
    # cmd = 'python run_model.py' + ' --data_path ' + site_name + '.csv'
    # print(cmd)
    # p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # for line in p.stdout.readlines():
    #     print(line)
    # retval = p.wait()



def save_data(now_time, site_name):
    now_time_utc = pd.to_datetime(now_time, unit='s')
    near_minute = np.floor(now_time_utc.minute / time_freq) * time_freq
    end_time_utc = now_time_utc.replace(minute=near_minute.astype(int), second=0, microsecond=0)
    end_time_unix = calendar.timegm(end_time_utc.timetuple())

    loaddata = np.load(
        './results/informer_JFNG_data_15min_unwind_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/real_prediction_{}.npy'.format(
            site_name))
    data_log = pd.DataFrame(loaddata[:, :, 0], index=['prediction_rainfall'])

    for i in range(0, data_log.shape[1]):
        if isnan(data_log[i].item()):
            data_log.iloc[0, i] = 0
    data_log.insert(data_log.shape[1], 'time', pd.to_datetime(now_time, unit='s'))
    data_log.to_csv(f'./log/{site_name}_prediction_log.csv', mode='a')


# 定时系统
def job(site_name, now_time):  # 定时任务
    time_interval = seq_len / 4 * 60 * 60  # 15小时的历史数据 = seqlen=60
    history_time = now_time - time_interval - 10 * 60  # 15小时的历史数据 = seqlen=60; 10*60: 10分钟预留空间
    print('当前时间(CST)：' + time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(now_time)))  # 当前时间
    print('数据获取起点时间(CST)：' + time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(history_time)))  # 获取数据起点时间
    flag = get_data(now_time, history_time, site_name)
    # flag = 1
    if flag == 1:
        run_model(site_name)
        save_data(now_time, site_name)
        print('Successfully Done')
    elif flag == -1:
        print("存在空表")
    else:
        print('No valid data')


# def get_devices():
#     db_config = {
#         'host': mysql_host,  # 替换为您的 MySQL 主机地址
#         'user': mysql_user,  # 替换为您的用户名
#         'password': mysql_password,  # 替换为您的密码
#         'database': mysql_database,  # 替换为您的数据库名
#         'port': mysql_port,  # 如果使用非默认端口，请修改
#     }
#     try:
#         # 建立数据库连接
#         cnx = mysql.connector.connect(**db_config)
#         cursor = cnx.cursor()
#         # 执行 SQL 查询
#         query = f"SELECT {mysql_listname} FROM {mysql_table}"
#         cursor.execute(query)
#         # 获取所有结果
#         results = cursor.fetchall()
#         # 提取以 'b' 开头的设备编码
#         b_devices = [row[0] for row in results if isinstance(row[0], str) and row[0].startswith('b')]
#
#         return b_devices
#
#     except Exception as e:
#         print(f"MySQL设备获取处理失败: {str(e)}")


def get_env_variable(key: str, required: bool = True) -> str:
    """
    从环境变量中获取指定的键值。
    如果 required=True 且未设置，则打印错误并退出。
    """
    val = os.getenv(key)
    if required and (val is None or val.strip() == ""):
        print(f"[ERROR] 环境变量 `{key}` 未设置或为空。")
        sys.exit(1)
    return val.strip() if val else ""


if __name__ == "__main__":
    # 全局变量
    global time_freq, seq_len, site_name
    seq_len = 48 + 12  # 序列长度，如果生成csv的行数少于这个值就会报错，有nan也会造成计算错误
    count_n = 60
    time_freq = 15
    # 获取环境变量 数据库信息
    clickhouse_host = get_env_variable("CLICKHOUSE_HOST", required=False) or "bytehouse.huoshan.accurain.cn"
    clickhouse_port = get_env_variable("CLICKHOUSE_PORT", required=False) or "80"
    clickhouse_user = get_env_variable("CLICKHOUSE_USER", required=False) or "accurain_guest"
    clickhouse_password = get_env_variable("CLICKHOUSE_PASSWORD", required=False) or "V3VuWS7%FWs@u"
    # 设备列表获取的mysql
    # mysql_host = get_env_variable("MYSQL_HOST", required=False) or "mysqlbd18543afbe5.rds.ivolces.com"
    # mysql_port = get_env_variable("MYSQL_PORT", required=False) or "3306"
    # mysql_user = get_env_variable("MYSQL_USER", required=False) or "gnss"
    # mysql_password = get_env_variable("MYSQL_PASSWORD", required=False) or "klv!dopY$uN8I"
    # mysql_table = get_env_variable("MYSQL_TABLE", required=False) or "gnss"
    # mysql_listname = get_env_variable("MYSQL_LISTNAME", required=False) or "sb_jbxx"
    # mysql_database = get_env_variable("MYSQL_DATABASE", required=False) or "NUM"
    # 执行的时间戳
    # exec_timestamp = int(get_env_variable("EXEC_TIMESTAMP"))
    # 解析设备列表
    # devices = get_devices()

    # 测试用
    exec_timestamp = int('1748999700')
    devices = ['b04']

    # 顺序执行设备处理
    success_count = 0
    for device in devices:
        try:
            print(f"=== 开始处理设备 {device} ===")
            job(device.strip(), exec_timestamp)
            success_count += 1
        except Exception as e:
            print(f"!!! 设备 {device} 处理失败: {str(e)}")
    # 最终状态判断（至少一个成功即视为任务成功）
    if success_count == 0:
        print("所有设备处理失败！")
        sys.exit(1)
    else:
        print(f"成功处理 {success_count}/{len(devices)} 台设备")
        sys.exit(0)
