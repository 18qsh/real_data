# import argparse
# import os
# import torch
# import glob
# from tqdm import tqdm
#
# from exp.exp_informer import Exp_Informer
#
# # update 2024年10月21日09:12:12
#
# folder_path = './tmp/B70'  # 只需要修改这里
# extension = '.csv'
#
# parser = argparse.ArgumentParser(description='数据导出')
# parser.add_argument('--file_path', type=str, help='csv文件夹路径')
#
# args = parser.parse_args()
#
# if args.file_path:
#     folder_path = args.file_path
#
# csv_files = glob.glob(os.path.join(folder_path, f'*{extension}'))
#
# dir2=[]
# for file_path in csv_files:
#     dir1 = os.path.basename(file_path)
#     dir2.append(dir1);
#     # dir2 = [dir1]
# # print(dir2)
# # print(len(dir2))
#
# dir4=[]
# for file_path in csv_files:
#     dir3 = os.path.basename(file_path)
#     dir4.append(dir3[:-4])
# print(dir4)
# # print(type(dir4))
# # print(len(dir4))
#
# for i, j in tqdm(zip(dir2, dir4), total=len(dir2), desc="处理文件"):
#     parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
#
#     parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
#
#     parser.add_argument('--data', type=str, default='B70', help='data')
#     parser.add_argument('--root_path', type=str, default=folder_path, help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default=i, help='data file')
#     parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='tp', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='15t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#
#     parser.add_argument('--seq_len', type=int, default=24, help='input sequence length of Informer encoder')
#     parser.add_argument('--label_len', type=int, default=12, help='start token length of Informer decoder')
#     parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
#     # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
#
#     parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=1, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
#     parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
#     parser.add_argument('--padding', type=int, default=0, help='padding type')
#     parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
#     parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
#     parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
#     parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu',help='activation')
#     parser.add_argument('--file_path', type=str, help='file path of the input data')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
#     parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data')
#     parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
#     parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
#     parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='exp_3',help='exp description')
#     parser.add_argument('--loss', type=str, default='mse',help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#     parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
#
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
#
#     args = parser.parse_args()
#
#     # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#     # 强制使用CPU
#     args.use_gpu = False
#
#     # if args.use_gpu:
#     #     args.gpu = 0  # 默认使用第一个GPU（索引0）
#     #     args.device_ids = [0]  # 单个GPU设备ID列表
#     #     args.use_multi_gpu = False  # 强制关闭多GPU模式
#
#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ','')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]
#
#     # Exp = Exp_Informer
#     # exp = Exp_Informer(args)
#     # if args.data not in exp._get_data(exp.args, True).data_dict.key():
#     #     exp._get_data(exp.args, True).data_dict.update({j:"Dataset_Custom"})
#     #     print(Exp_Informer._get_data().data_dict)
#
#
#     data_parser = {
#         'hkws_2_5':{'data':'hkws_2_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
#         'era5_2016_5':{'data':'era5_2016_5.csv','T':'tp','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
#         'YLJ01_22':{'data':'YLJ01_22.csv','T':'tp','M':[1,1,1],'S':[1,1,1],'MS':[1,1,1]},
#         'hkws_3_5':{'data':'hkws_3_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
#         'YLJ01_5_1':{'data':'YLJ01_5_1.csv','T':'tp','M':[1,1,1],'S':[1,1,1],'MS':[1,1,1]},
#         # 'hkws_4_5':{'data':'hkws_4_5.csv','T':'tp','M':[5,5,5],'S':[1,1,1],'MS':[5,5,1]},
#         'JFNG_data_15min': {'data': i, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
#         'JFNG_data_15min_unwind': {'data': args.data_path, 'T': 'tp', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
#         'JFNG_pwvtp_15min': {'data': i, 'T': 'tp', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
#         'JFNG_data_1h': {'data': i, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
#     }
#
#     if args.data in data_parser.keys():
#         data_info = data_parser[args.data]
#         args.data_path = data_info['data']
#         args.target = data_info['T']
#         args.enc_in, args.dec_in, args.c_out = data_info[args.features]
#
#     # add data into data_parser
#     if args.data not in data_parser.keys():
#         data_parser.update({j:{ 'data':i, 'T': args.target, 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},} )
#         if args.data in data_parser.keys():
#             data_info = data_parser[args.data]
#             args.data_path = data_info['data']
#             args.target = data_info['T']
#             args.enc_in, args.dec_in, args.c_out = data_info[args.features]
#     # print(data_parser)
#
#     args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
#     args.detail_freq = args.freq
#     args.freq = args.freq[-1:]
#
#     print('Args in experiment:')
#     print(args)
#
#     Exp = Exp_Informer
#
#     for ii in range(args.itr):
#         # setting record of experiments
#         setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
#                                                                                                              args.data,
#                                                                                                              args.features,
#                                                                                                              args.seq_len,
#                                                                                                              args.label_len,
#                                                                                                              args.pred_len,
#                                                                                                              args.d_model,
#                                                                                                              args.n_heads,
#                                                                                                              args.e_layers,
#                                                                                                              args.d_layers,
#                                                                                                              args.d_ff,
#                                                                                                              args.attn,
#                                                                                                              args.factor,
#                                                                                                              args.embed,
#                                                                                                              args.distil,
#                                                                                                              args.mix,
#                                                                                                              args.des,
#                                                                                                              ii)
#
#         exp = Exp(args)  # set experiments
#         if args.do_predict:
#             print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#             exp.predict(setting, True)
#
#         torch.cuda.empty_cache()
#
#     i =+ 1
#     j =+ 1

#
import argparse
import os
import torch
import glob
from tqdm import tqdm
from exp.exp_informer import Exp_Informer


def run(folder_path, extension='.csv'):
    """
    主运行函数，处理指定文件夹下的CSV文件并执行预测

    参数:
        folder_path (str): CSV文件所在文件夹路径
        extension (str): 要查找的文件扩展名
    """
    parser = argparse.ArgumentParser(description='数据导出')
    parser.add_argument('--file_path', type=str, help='csv文件夹路径')
    args = parser.parse_args()

    if args.file_path:
        folder_path = args.file_path

    csv_files = glob.glob(os.path.join(folder_path, f'*{extension}'))

    dir2 = [os.path.basename(file_path) for file_path in csv_files]
    dir4 = [os.path.basename(file_path)[:-4] for file_path in csv_files]
    print(dir4)

    for i, j in tqdm(zip(dir2, dir4), total=len(dir2), desc="处理文件"):
        # 设置参数解析器
        parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

        # 添加所有参数
        parser.add_argument('--model', type=str, default='informer',
                            help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
        parser.add_argument('--data', type=str, default=f'B93', help='data')
        parser.add_argument('--root_path', type=str, default=folder_path, help='root path of the data file')
        parser.add_argument('--data_path', type=str, default=i, help='data file')
        parser.add_argument('--features', type=str, default='MS',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='tp', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='15t',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
        parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
        parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=1, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.01, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--file_path', type=str, help='file path of the input data')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_false',
                            help='whether to predict unseen future data', default=True)
        parser.add_argument('--mix', action='store_false',
                            help='use mix attention in generative decoder', default=True)
        parser.add_argument('--cols', type=str, nargs='+',
                            help='certain cols from the data files as the input features')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='exp_a', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true',
                            help='use automatic mixed precision training', default=False)
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        args = parser.parse_args()

        # 强制使用CPU
        # args.use_gpu = False
        if args.use_gpu:
            args.gpu = 0  # 默认使用第一个GPU（索引0）
            args.device_ids = [0]  # 单个GPU设备ID列表
            args.use_multi_gpu = False  # 强制关闭多GPU模式
        # if args.use_gpu and args.use_multi_gpu:
        #     args.devices = args.devices.replace(' ', '')
        #     device_ids = args.devices.split(',')
        #     args.device_ids = [int(id_) for id_ in device_ids]
        #     args.gpu = args.device_ids[0]

        # 数据解析字典
        data_parser = {
            'hkws_2_5': {'data': 'hkws_2_5.csv', 'T': 'tp', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
            'era5_2016_5': {'data': 'era5_2016_5.csv', 'T': 'tp', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'YLJ01_22': {'data': 'YLJ01_22.csv', 'T': 'tp', 'M': [1, 1, 1], 'S': [1, 1, 1], 'MS': [1, 1, 1]},
            'hkws_3_5': {'data': 'hkws_3_5.csv', 'T': 'tp', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
            'YLJ01_5_1': {'data': 'YLJ01_5_1.csv', 'T': 'tp', 'M': [1, 1, 1], 'S': [1, 1, 1], 'MS': [1, 1, 1]},
            'JFNG_data_15min': {'data': i, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
            'JFNG_data_15min_unwind': {'data': args.data_path, 'T': 'tp', 'M': [5, 5, 5], 'S': [1, 1, 1],
                                       'MS': [5, 5, 1]},
            'JFNG_pwvtp_15min': {'data': i, 'T': 'tp', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
            'JFNG_data_1h': {'data': i, 'T': 'tp', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        }

        # 更新数据参数
        if args.data in data_parser.keys():
            data_info = data_parser[args.data]
            args.data_path = data_info['data']
            args.target = data_info['T']
            args.enc_in, args.dec_in, args.c_out = data_info[args.features]

        # 添加新数据到data_parser
        if args.data not in data_parser.keys():
            data_parser.update({j: {'data': i, 'T': args.target, 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [5, 5, 1]}})
            if args.data in data_parser.keys():
                data_info = data_parser[args.data]
                args.data_path = data_info['data']
                args.target = data_info['T']
                args.enc_in, args.dec_in, args.c_out = data_info[args.features]

        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        print('Args in experiment:')
        print(args)

        Exp = Exp_Informer

        for ii in range(args.itr):
            # 设置实验记录
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn,
                args.factor, args.embed, args.distil, args.mix, args.des, ii)

            exp = Exp(args)  # 创建实验
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

#
# if __name__ == '__main__':
#     # 默认参数
#     devices = 'B92'
#     folder_path = f'./tmp/{devices}'  # 默认文件夹路径
#     extension = '.csv'  # 默认文件扩展名
#
#     # 运行主函数
#     run(folder_path, extension)