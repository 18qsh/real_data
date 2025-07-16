import os
import sys
from pathlib import Path


def run(
        input_csv: str,
        site_name: str,
        start_time: str,
        end_time: str,
        # output_path: str = None,
        informer_folder: str = None,
        informer_params: dict = None  # 传递给 test02_copy.run() 的参数
):
    script_dir = "D:/app_data/pycharm_data/real_data"
    if script_dir not in sys.path:
        sys.path.append(script_dir)

    try:
        import pred_data_process
        import test02_copy

        # 1. 分割CSV文件
        pred_data_process.split_csv(file=input_csv, site_name=site_name)

        # 2. 运行Informer模型
        informer_input_folder = informer_folder or f"./tmp/{site_name}"

        # 提供默认参数
        default_params = {
            "folder_path": informer_input_folder,
            "extension": ".csv",
        }
        params = {**default_params, **(informer_params or {})}

        # 调用 test02_copy.run() 并传参
        test02_copy.run(**params)

        # 4. 合并结果
        output_path =f"./pred_output/{site_name}.csv"
        print(f"[Step 3/3] Merging predictions to: {output_path}")
        pred_data_process.merge_csv(
            site_name=site_name,
            start_time=start_time,
            end_time=end_time,

        )

        print(f"[Success] Pipeline completed! Results saved to: {output_path}")
        return True

    except Exception as e:
        print(f"[Error] Pipeline failed: {str(e)}")
        return False

