#!/bin/bash

site_name="B187"
start_date="2024-05-31"
end_date="2025-05-31"

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
echo "���� ��ȡ���� �ű�..."
get_db_file="./tmp/${site_name}.csv"
rm -rf $get_db_file
python get_data.py --name $site_name --start $start_date --end $end_date


#echo "���� �ָ����� �ű�..."
#rm -rf ./tmp/"$site_name"
#python pred_data_process.py --name $site_name --file $get_db_file --main split --start_time $start_date --end_time $end_date
#
#echo "���� Ԥ��ģ�� �ű�..."
#find ./results/informer_JFNG_data_15min_unwind_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/ -maxdepth 1 -type f -name "real_prediction_${site_name}_*" -print0 | xargs -0 rm -f
#python test02_copy.py --file_path ./tmp/"$site_name"
#
#echo "���� ������� �ű�..."
#python pred_data_process.py --name $site_name --file 'None' --main merge --start_time $start_date --end_time $end_date
#
#echo "����������"

# �˳����⻷��
deactivate

echo "�ű�ִ�����"
