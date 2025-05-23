"""
主函数
"""
from gensim.models import KeyedVectors
import numpy as np
from data_get import data_get, build_dict, convert_corpus_to_id, IMDBDataset, collate_fn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from train import train_model
import math

# 转换到GPU模式
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 获取数据并用训练集数据生成词典
train_data = data_get('train')
vaild_data = data_get('valid')
word2id_dict = build_dict(train_data)

# 将数据根据字典转换为ID
train_dataset = convert_corpus_to_id(train_data, word2id_dict)
vaild_dataset = convert_corpus_to_id(vaild_data, word2id_dict)

# 生成数据类和数据加载器
train_data_set = IMDBDataset(train_dataset)
vaild_data_set = IMDBDataset(vaild_dataset)
train_dataloader = DataLoader(
    train_data_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
vaild_dataloader = DataLoader(
    vaild_data_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
data_Loaders = {
    'train': train_dataloader,
    'valid': vaild_dataloader
}

# 加载预训练的词向量模型
word2vec_model = KeyedVectors.load_word2vec_format(
    'C:\\Users\\hcjtx\\PycharmProjects\\IMDB情感分析\\GoogleNews-vectors-negative300.bin', binary=True)
embedding_dim = 300  # 词向量维度

# 构建嵌入矩阵
embedding_matrix = np.zeros((len(word2id_dict), embedding_dim))  # 初始化嵌入矩阵

# 词向量模型载入嵌入矩阵
for word, idx in word2id_dict.items():
    if word == '[pad]':
        continue
    elif word in word2vec_model:
        embedding_matrix[idx] = word2vec_model[word]
    else:
        # 处理未登录词随机初始化
        embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

# 嵌入矩阵转换为 PyTorch 张量
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

# 定义模型


class SelfAttention(nn.Module):
    """
    自定义多头注意力层
    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 词向量的维度
        self.heads = heads  # 多头注意力的头数
        self.head_dim = embed_size // heads  # 每个头要处理的维度

        # 定义线性层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):  # (bitch_size, seq_len, embed_size)
        bitch_size = query.shape[0]  # 批次大小
        # 句子的长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分头降维
        values = values.reshape(bitch_size, value_len,
                                self.heads, self.head_dim)
        keys = keys.reshape(bitch_size, key_len,
                            self.heads, self.head_dim)
        queries = query.reshape(bitch_size, query_len,
                                self.heads, self.head_dim)

        # 进行线性转换投影到另个空间分别学习
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力分数
        # 调整维度，适应矩阵乘积
        # -> (N, heads, query_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)  # -> (N, heads, key_len, head_dim)
        # 计算点积：(N, heads, query_len, head_dim) @ (N, heads, head_dim, key_len)
        # -> (N, heads, query_len, key_len)
        energy = torch.matmul(queries, keys.transpose(-2, -1))

        # pad掩码处理
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(
            energy / (self.embed_size ** (1/2)), dim=3)  # 注意力分数归一化获得权重
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            bitch_size, query_len, self.heads*self.head_dim
        )  # 与values进行矩阵乘法，得到输出并将多个头进行简单拼接，再放入全连接层训练
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape (N, query_len, heads, head_dim)

        out = self.fc_out(out)  # 将结果映射回原始嵌入维度
        return out


class encoderBlock(nn.Module):
    """
    encoder块,包含多头注意力层,全连接层和残差连接结构
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(encoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)  # 多头注意力层
        self.norm1 = nn.LayerNorm(embed_size)  # 层归一化，后续对注意力分数归一化
        self.norm2 = nn.LayerNorm(embed_size)  # 层归一化，后续对前向传播结果归一化
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),  # 引入非线性特性
            nn.Linear(forward_expansion*embed_size, embed_size)
        )  # 前向传播层
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)  # 多头注意力

        x = self.dropout(self.norm1(attention + query))  # 先残差结果再层归一化在dropout
        forward = self.feed_forward(x)  # 前向传播
        out = self.dropout(self.norm2(forward + x))  # 先残差结果再层归一化在dropout
        return out


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, embed_size, max_len=512, device='cuda'):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size).to(device)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float(
        ) * (-math.log(10000.0) / embed_size)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_size)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Encoder(nn.Module):
    def __init__(self,
                 embed_size,  # 词向量维度
                 num_layers,  # 编码器层数
                 heads,  # 多头注意力头数
                 device,  # 设备
                 forward_expansion,  # 前向传播扩展倍数
                 dropout,
                 max_length  # 最大长度
                 ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=1)  # 词嵌入
        self.position_embedding = PositionalEncoding(
            embed_size, max_length, device=device)  # 位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_size))  # 分类向量
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 初始化分类向量
        self.layers = nn.ModuleList(
            [
                encoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)]
        )  # 多层编码器层
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        base_mask = (src != 1)  # 生成掩码
        cls_mask = torch.ones(base_mask.size(
            0), 1, dtype=torch.bool, device=base_mask.device)  # (N, 1)
        src_mask = torch.cat([cls_mask, base_mask], dim=1).unsqueeze(
            1).unsqueeze(2)  # (N, 1,1,src_len+1)
        return src_mask.to(self.device)

    def forward(self, x):
        out = self.word_embedding(x)  # 进行词嵌入
        cls_token = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_token, out), dim=1)  # 添加分类向量
        out = self.position_embedding(out)  # 进行位置嵌入
        mask = self.make_src_mask(x)  # 生成掩码
        for layer in self.layers:
            out = layer(out, out, out, mask)  # 多层编码器层

        return out


class TransformerClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(TransformerClassifier, self).__init__()
        self.encoder = encoder  # 已定义好的 Transformer Encoder

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(encoder.embed_size, encoder.embed_size),  # 全连接层
            nn.Tanh(),  # 非线性激活
            nn.Linear(encoder.embed_size, num_classes)  # 输出
        )

    def forward(self, x):
        # 编码器输出 (N, seq_len + 1, embed_size)
        out = self.encoder(x)

        # 取出 [CLS] token
        cls_output = out[:, 0, :]

        # 分类预测 (N, num_classes)
        logits = self.classifier(cls_output)

        return logits


# 实例化模型
model = TransformerClassifier(
    encoder=Encoder(
        embed_size=300,       # 词向量维度
        num_layers=4,         # 编码器层数
        heads=5,             # 多头注意力头数
        device=device,        # 设备（GPU/CPU）
        forward_expansion=4,  # 前向传播扩展倍数
        dropout=0.1,
        max_length=5000       # 最大序列长度
    ),
    num_classes=2  # 分类数
)


# 优化器设置
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# 学习率衰减策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 损失值计算
loss_of_model = torch.nn.CrossEntropyLoss()
# 保存文件
# filename = 'IMDBClassification/save_model/best_model.pt'
# 训练次数
num_epoch = 20

# 训练模型
model, val_acc_history, train_acc_history, valid_loss, train_loss = train_model(
    model, loss_of_model, optimizer, scheduler, num_epoch, data_Loaders, device)
