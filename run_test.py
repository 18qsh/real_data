from sh import run

# 定义 test02_copy.run() 的参数
success = run(
    input_csv="./tmp/B93.csv",
    site_name="B93",
    start_time="2025-05-08",
    end_time="2025-06-25",
    informer_params={
    "folder_path": "./tmp/B93",
    "extension": ".csv",
    }
)


# import subprocess
#
# venv_python = "D:/app/python/python.exe"
# script_dir = "D:/app_data/pycharm_data/real_data(1)"
#
# # 运行各个Python脚本
# # subprocess.run([venv_python, "pred_data_process.py", "--name", "B15", "--file", "./tmp/B15.csv", "--main", "split", "--start_time", "2025-05-13", "--end_time", "2025-06-17"], cwd=script_dir)
# # subprocess.run([venv_python, "test02_copy.py", "--file_path", "./tmp/B15"], cwd=script_dir)
# subprocess.run([venv_python, "pred_data_process.py", "--name", "B91", "--file", "None", "--main", "merge", "--start_time", "2025-04-08", "--end_time", "2025-06-25"], cwd=script_dir)
