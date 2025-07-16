#!/bin/bash

site_name="B15"
start_date="2025-05-14"
end_date="2025-06-17"
0
# ���ô���ʱ�˳�
set -e

# �������⻷��·��
VENV_PATH="/home/ubuntu/accurain_predict/env_readdata"

# ������⻷���Ƿ����
if [ ! -d "$VENV_PATH" ]; then
    echo "�������⻷��������: $VENV_PATH"
    exit 1
fi

# �������⻷��
echo "�������⻷��..."
source "$VENV_PATH/bin/activate"

# ������⻷���Ƿ���ȷ����
if [ -z "$VIRTUAL_ENV" ]; then
    echo "�������⻷������ʧ��"
    exit 1
fi

# ��ӡ��ǰ Python ·����ȷ��
echo "ʹ�õ� Python ������: $(which python)"

echo $site_name

cd /home/ubuntu/accurain_predict/GPU_model_predict
# ���� Python �ű�

get_db_file="./tmp/${site_name}.csv"

echo "���� �ָ����� �ű�..."
rm -rf ./tmp/"$site_name"
python pred_data_process.py --name $site_name --file $get_db_file --main split --start_time $start_date --end_time $end_date

echo "���� Ԥ��ģ�� �ű�..."
find ./results/informer_B112_ftMS_sl24_ll12_pl4_dm512_nh8_el3_dl3_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_3_0/ -maxdepth 1 -type f -name "real_prediction_${site_name}_*" -print0 | xargs -0 rm -f
python test02_copy.py --file_path ./tmp/"$site_name"

echo "���� ������� �ű�..."
python pred_data_process.py --name $site_name --file 'None' --main merge --start_time $start_date --end_time $end_date

echo "����������"

# �˳����⻷��
deactivate

echo "�ű�ִ�����"
