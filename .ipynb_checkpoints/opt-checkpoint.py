import torch
# 模型配置文件

# 优化器参数
LEARNING_RATE=1e-3
# LEARNING_RATE =2e-4
WEIGHT_DECAY=1e-2
# WEIGHT_DECAY=1e-4

# 损失函数设置
ALPHA=0.75
GAMMA=2

# 训练设置
batch_size=512
epochs = 300
number_workers=0
# 梯度累加
accumulation_step=1
device="cuda" if torch.cuda.is_available() else "cpu"
optimizer="adamw"
# warmup 配合余弦退火，其他输入都是使用ReduceLROnPlateau自适应学习率
lrsc="warmup"
# 损失函数Focal和CE
loss="Focal"
# LSTM输入序列长度
time_tri=1
# 数据集路径
data_dir="./data/example/data.csv"
# 保存结果目录
save_dir="./evaluate/"
# 数据集的名字，影响权重读取和保存的名字
filename="dataname"

# 模型控制
# 权重保存
keep=True
# 权重读取
resume=False
# 读取checkpoint
resume_path="./weights/GANBILSTM.pth"
# 网络结构
# 嵌入特征数目
emb_size=27
# 隐藏层
rnn_size=256
rnn_layers=1
dropout=0.3
# 使用GAN模块
useGAN=True
# 模型名字，保存权重文件相关
NAME="GANBILSTM"
# loader配置
# testRatio表示了训练集和测试集的比例，验证集是从训练集中划分的，valRatio代表这个比例
testRatio=0.2
valRatio=0.2
# 预读取GAN权重
useGAN_weights=False
GAN_path="./weights/ganmodel.pkl"
# CTGAN训练轮数
GAN_epoch=50
