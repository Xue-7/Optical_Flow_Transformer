import os
from Transformer import *
import data_loader
import data_loader_classify
from torch.utils.data import DataLoader
from train import *
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#随机数种子确定
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

#use_gpu = torch.cuda.is_available()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#参数设置
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--enc_input_size", type=int, default=2051, help="encoder input size")
parser.add_argument("--dec_output_size", type=int, default=1, help="decoder input size")
parser.add_argument("--output_size", type=int, default=1, help="预测步长")
parser.add_argument("--num_epochs", type=int, default=5, help="epoch to start training from")
parser.add_argument("--num_layers", type=int, default=2, help="编码器和解码器的层数，这里两者层数相同，也可以不同")
parser.add_argument("--dropout", type=float, default=0.05, help="所有层的droprate都相同，也可以不同")
parser.add_argument("--batch_size", type=int, default=8, help="index of gpu")
parser.add_argument("--factor", type=int, default=1, help="学习率因子")
parser.add_argument("--warmup", type=int, default=2000, help="学习率上升步长")
parser.add_argument("--ctx", type=str, default=device, help="index of gpu")
parser.add_argument("--num_hiddens", type=int, default=10, help="隐层单元的数目——表示FFN中间层的输出维度")
parser.add_argument("--num_heads", type=int, default=8, help="attention的数目")
parser.add_argument("--d_model", type=int, default=512, help='dimension of model')
parser.add_argument("--lr", type=float, default=0.005, help="学习率")
parser.add_argument("--shuffle", type=bool, default=True, help="学习率")
parser.add_argument("--station_num", type=int, default=0, help="站点")
parser.add_argument("--time_step", type=int, default=0, help="预测时间步")
parser.add_argument("--xtw", type=int, default=10, help="预测时间步")
parser.add_argument("--ytw", type=int, default=8, help="预测时间步")
opt = parser.parse_args()


#src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size, num_steps)
# 读取训练测试数据
dataset_train = data_loader.ModelRadarDataset(mode='train', station=opt.station_num)
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=opt.shuffle)
dataset_test = data_loader.ModelRadarDataset(mode='test', station=opt.station_num)
dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
dataset_val = data_loader.ModelRadarDataset(mode='val',station=opt.station_num)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False)
# dataset_train = data_loader_classify.ModelRadarDataset(mode='train', station=opt.station_num)
# dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=opt.shuffle)
# dataset_test = data_loader_classify.ModelRadarDataset(mode='test', station=opt.station_num)
# dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
# dataset_val = data_loader_classify.ModelRadarDataset(mode='val',station=opt.station_num)
# dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False)
#adj = graph.normalize_adj(adj)

encoder = TransformerEncoder(input_size=opt.d_model,
                             n_layers=opt.num_layers,
                             ffn_hidden_size=opt.num_hiddens,
                             num_heads=opt.num_heads,
                             dropout=opt.dropout, )
decoder = TransformerDecoder(output_size=opt.output_size,
                             input_size=opt.d_model,
                             n_layers=opt.num_layers,
                             ffn_hidden_size=opt.num_hiddens,
                             num_heads=opt.num_heads,
                             dropout=opt.dropout, )


class transformer(nn.Module):
    def __init__(self, enc_net, dec_net):
        super(transformer, self).__init__()
        self.enc_net = enc_net  # TransformerEncoder的对象
        self.dec_net = dec_net  # TransformerDecoder的对象

        self.dense1 = nn.Linear(2051,1)
        self.dense2 = nn.Linear(2051,2050)


    def forward(self, enc_X,enc_X_FM, dec_X, valid_length=None, max_seq_len=None):
        """
        enc_X: 编码器的输入
        dec_X: 解码器的输入
        valid_length: 编码器的输入对应的valid_length,主要用于编码器attention的masksoftmax中，
                      并且还用于解码器的第二个attention的masksoftmax中
        max_seq_len:  位置编码时调整sin和cos周期大小的，默认大小为enc_X的第一个维度seq_len
        """

        # 1、通过编码器得到编码器最后一层的输出enc_output
        # x = self.dense1(enc_X.permute(0,2,1)).permute(0,2,1)
        # fm = self.dense2(enc_X.permute(0,2,1)).permute(0,2,1)
        enc_output = self.enc_net(enc_X,enc_X_FM, valid_length, max_seq_len)
        # 2、state为解码器的初始状态，state包含两个元素，分别为[enc_output, valid_length]
        state = self.dec_net.init_state(enc_output, valid_length)
        # 3、通过解码器得到编码器最后一层到线性层的输出output，这里的output不是解码器最后一层的输出，而是
        #    最后一层再连接线性层的输出
        output = self.dec_net(dec_X, state)
        return output

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('all_30min_final_adaptive.csv')
station = 0
data = np.array(df.iloc[:, station+3:station+4])
# data = np.array(df.iloc[:, station+4:station+5])
#mean, std = np.mean(data), np.std(data)
#data = (data - mean) / std
# data = data/100
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

#data = graph.normalize_feat(data)

mode = ['test','train']
def mean_absolute_percentage_error(y_true, y_pred):
    res = []
    for i in range(len(y_pred)):
        if y_true[i] < 1e-4:
            continue
        a = np.abs((y_pred[i] - y_true[i]) / (y_true[i] + 1e-9))
        res.append(a)
    return np.mean(res)

model = transformer(encoder, decoder)

time_str = arrow.now().format('YYYYMMDD_HHmmss')
# model保存路径设置
model_path = Path(f'model/_-{time_str}')
model_path.mkdir(exist_ok=True)
# 日志文件
log_file = model_path / Path('train.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')  # level为日志等级
logger = logging.getLogger('main')  # 初始化
logger.setLevel(logging.INFO)
logger.info(opt)

if mode[0] == 'train':
    model.train()
    pre0, true0 = train(model, dataloader_train, opt.lr, opt.factor, opt.warmup, opt.num_epochs, device, model_path, logger,dataloader_val)
    a = pre0.clone().cpu().detach().numpy().reshape(-1,opt.output_size)
    b = true0.clone().cpu().detach().numpy().reshape(-1,opt.output_size)
    mae = mean_absolute_error(b, a)
    print('Train:|mae值为:', mae)


    pre = np.hstack([scaler.inverse_transform(a[:,i].reshape(-1,1)) for i in range(opt.output_size)])
    true = np.hstack([scaler.inverse_transform(b[:,i].reshape(-1,1)) for i in range(opt.output_size)])

    mae = mean_absolute_error(true, pre )
    print('Train:|mae值为:', mae)
    logger.info(f'Train:|MAE:{mae}')

    plt.cla()
    plt.plot(true[-200:, opt.time_step], label='true')
    plt.plot(pre[-200:, opt.time_step], label='pred')
    plt.xlabel('day')
    plt.ylabel('prediction')
    name = f'plot/epoch{str(opt.num_epochs)}MAE{str(mae)}station{opt.station_num}train.png'
    plt.savefig(name)
    #plt.show()

if mode[1]=='train':
    #model.eval()
    pre0, true0 = train(model, dataloader_test, opt.lr, opt.factor, opt.warmup, opt.num_epochs, device, model_path, logger,dataloader_test)
    a = pre0.cpu().detach().numpy().reshape(-1, opt.output_size)
    b = true0.cpu().detach().numpy().reshape(-1, opt.output_size)
    mae = mean_absolute_error(b, a )
    print('Test:mae值为:', mae)
    logger.info(f'Test:|MAE:{mae}')
    list = []
    list = np.hstack((a.reshape(-1, opt.output_size),b.reshape(-1, opt.output_size)))
    list_tabel = pd.DataFrame(data=list.tolist())
    list_tabel.to_csv(f'result/epoch{str(opt.num_epochs)}MAE{str(mae)}station{opt.station_num}0.csv')

    pre = np.hstack([scaler.inverse_transform(a[:,i].reshape(-1,1)) for i in range(opt.output_size)])
    true = np.hstack([scaler.inverse_transform(b[:,i].reshape(-1,1)) for i in range(opt.output_size)])

    print(pre)
    mae = mean_absolute_error(true, pre )
    print('Test:mae值为:', mae)
    logger.info(f'Test:|MAE:{mae}')
    list = np.hstack((pre.reshape(-1, opt.output_size),true.reshape(-1, opt.output_size)))
    list_tabel = pd.DataFrame(data=list.tolist())
    list_tabel.to_csv(f'result/epoch{str(opt.num_epochs)}MAE{str(mae)}station{opt.station_num}.csv')

    plt.cla()
    plt.plot(true[-200:, opt.time_step], label='true')
    plt.plot(pre[-200:, opt.time_step], label='pred')
    plt.xlabel('day')
    plt.ylabel('prediction')
    name = f'plot/epoch{str(opt.num_epochs)}MAE{str(mae)}station{opt.station_num}test.png'
    plt.savefig(name)
    #plt.show()