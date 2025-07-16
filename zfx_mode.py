import argparse
import os
import torch
import glob
import re
import sys

from exp.exp_informer import Exp_Informer

# zfx update start

def find_matching_model_directory(checkpoints_dir, model_id):
    """
    在checkpoints目录中查找匹配特定模型ID的目录
    
    Args:
        checkpoints_dir: 模型检查点目录
        model_id: 模型ID (例如: B16)
    
    Returns:
        匹配的模型目录路径，如果没有找到则返回None
    """
    # 获取所有可能的模型目录
    all_model_dirs = glob.glob(os.path.join(checkpoints_dir, "informer_*"))
    
    # 正则表达式匹配模型ID
    pattern = rf"informer_({model_id})_"
    
    for model_dir in all_model_dirs:
        dir_name = os.path.basename(model_dir)
        # 使用精确匹配确保我们找到的是正确的模型ID
        # 例如B16不会匹配到B160
        match = re.match(pattern, dir_name)
        if match and match.group(1) == model_id:
            return model_dir
    
    return None

def extract_model_id_from_data_path(data_path):
    """
    从data_path参数中提取模型ID
    
    Args:
        data_path: 数据文件路径 (例如: B16.csv)
    
    Returns:
        模型ID (例如: B16)
    """
    # 使用正则表达式提取模型ID，匹配格式如B16, B107等
    match = re.match(r"(B\d+)\.csv", data_path)
    if match:
        return match.group(1)
    return None

# zfx update end

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, default='JFNG_data_15min_unwind', help='data')
parser.add_argument('--root_path', type=str, default='./real_data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='B16.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='tp', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='15t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
parser.add_argument('--c_out', type=int, default=5, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Exp = Exp_Informer
# exp = Exp_Informer(args)
# if args.data not in exp._get_data(exp.args, True).data_dict.key():
#     exp._get_data(exp.args, True).data_dict.update({j:"Dataset_Custom"})
#     print(Exp_Informer._get_data().data_dict)


data_parser = {
    'hkws_2_5':{'data':'hkws_2_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
    'era5_2016_5':{'data':'era5_2016_5.csv','T':'tp','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'YLJ01_22':{'data':'YLJ01_22.csv','T':'tp','M':[1,1,1],'S':[1,1,1],'MS':[1,1,1]},
    'hkws_3_5':{'data':'hkws_3_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
    'YLJ01_5_1':{'data':'YLJ01_5_1.csv','T':'tp','M':[1,1,1],'S':[1,1,1],'MS':[1,1,1]},
    # 'hkws_4_5':{'data':'hkws_4_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
    'JFNG_data_15min': {'data': args.data_path, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
    'JFNG_data_15min_unwind': {'data': args.data_path, 'T': 'tp', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
    'JFNG_pwvtp_15min': {'data': args.data_path, 'T': 'tp', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
    'JFNG_data_1h': {'data': args.data_path, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

# add data into data_parser
if args.data not in data_parser.keys():
    data_parser.update({args.data:{ 'data':args.data_path, 'T': args.target, 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [5, 5, 1]},} )
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
# print(data_parser)

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

# zfx update start
# 提取模型ID
model_id = extract_model_id_from_data_path(args.data_path)
if not model_id:
    print(f"错误: 无法从数据路径 '{args.data_path}' 提取模型ID。请确保数据路径格式为 'BXX.csv'")
    sys.exit(1)

print(f"提取的模型ID: {model_id}")
# 查找匹配的模型目录
model_dir = find_matching_model_directory(args.checkpoints, model_id)
if model_dir:
    print(f"找到匹配的模型目录: {os.path.basename(model_dir)}")
    parts = model_dir.split('_')
    # 提取模型参数
    args.data = parts[1]
    args.features = parts[2].replace('ft', '')
    args.seq_len = int(parts[3].replace('sl', ''))
    args.label_len = int(parts[4].replace('ll', ''))
    args.pred_len = int(parts[5].replace('pl', ''))
    args.d_model = int(parts[6].replace('dm', ''))
    args.n_heads = int(parts[7].replace('nh', ''))
    args.e_layers = int(parts[8].replace('el', ''))
    args.d_layers = int(parts[9].replace('dl', ''))
    args.d_ff = int(parts[10].replace('df', ''))
    args.attn = parts[11].replace('at', '')
    args.factor = int(parts[12].replace('fc', ''))
    args.embed = parts[13].replace('eb', '')
    args.distil = True if parts[14].replace('dt', '') == 'True' else False
    args.mix = True if parts[15].replace('mx', '') == 'True' else False
    args.des = parts[16]
    args.itr = int(parts[17])+1



# zfx update end

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                          args.data,
                                                                                                          args.features,
                                                                                                          args.seq_len,
                                                                                                          args.label_len,
                                                                                                          args.pred_len,
                                                                                                          args.d_model,
                                                                                                          args.n_heads,
                                                                                                          args.e_layers,
                                                                                                          args.d_layers,
                                                                                                          args.d_ff,
                                                                                                          args.attn,
                                                                                                          args.factor,
                                                                                                          args.embed,
                                                                                                          args.distil,
                                                                                                          args.mix,
                                                                                                          args.des,
                                                                                                          ii)

    exp = Exp(args)  # set experiments
    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

# zfx update start
print("save npy file path!")
# 保存到文件
with open(".pred_file_path", "w") as f:
    f.write(setting)

# zfx update end