#!/bin/bash

site_name="B163"
start_date="2025-05-30"
end_date="2025-06-10"

# 设置错误时退出
set -e

# 定义虚拟环境路径
VENV_PATH="/home/ubuntu/accurain_predict/env_readdata"

# 检查虚拟环境是否存在
if [ ! -d "$VENV_PATH" ]; then
    echo "错误：虚拟环境不存在: $VENV_PATH"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 检查虚拟环境是否正确激活
if [ -z "$VIRTUAL_ENV" ]; then
    echo "错误：虚拟环境激活失败"
    exit 1
fi

# 打印当前 Python 路径以确认
echo "使用的 Python 解释器: $(which python)"

echo $site_name

cd /home/ubuntu/accurain_predict/GPU_model_predict
# 运行 Python 脚本
echo "运行 获取数据 脚本..."
get_db_file="./tmp/${site_name}.csv"
rm -rf $get_db_file
python get_data.py --name $site_name --start $start_date --end $end_date

exit 1

echo "运行 分割数据 脚本..."
rm -rf ./tmp/"$site_name"
python pred_data_process.py --name $site_name --file $get_db_file --main split --start_time $start_date --end_time $end_date

echo "运行 预测模型 脚本..."
find ./results/informer_B163_ftMS_sl24_ll12_pl4_dm512_nh8_el3_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_3_0/ -maxdepth 1 -type f -name "real_prediction_${site_name}_*" -print0 | xargs -0 rm -f
python test02_copy.py --file_path ./tmp/"$site_name" --seq_len 24 --label_len 12 --pred_len 4 --d_model 512 --n_heads 8 --e_layers 3 --d_layers 3 --d_ff 2048 --distil True --mix True --des 3

echo "运行 结果处理 脚本..."
python pred_data_process.py --name $site_name --file 'None' --main merge --start_time $start_date --end_time $end_date

echo "结果处理完成"

# 退出虚拟环境
deactivate

echo "脚本执行完成"
