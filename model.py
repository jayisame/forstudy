import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 词向量的维度
        self.heads = heads  # 多头注意力的头数
        self.head_dim = embed_size // heads  # 每个头要处理的维度

        assert (self.head_dim * heads ==
                embed_size),  "Embed size needs  to  be div by heads"  # 检查是否整除
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
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        # pad掩码处理
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # Fills elements of self tensor with value where mask is True

        attention = torch.softmax(
            energy / (self.embed_size ** (1/2)), dim=3)  # 注意力分数归一化获得权重
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            bitch_size, query_len, self.heads*self.head_dim
        )  # 与values进行矩阵乘法，得到输出并将多个头进行拼接
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)  # 将结果映射回原始嵌入维度
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)  # 多头注意力
        self.norm1 = nn.LayerNorm(embed_size)  # 层归一化，后续对注意力分数归一化
        self.norm2 = nn.LayerNorm(embed_size)  # 层归一化，后续对前向传播结果归一化
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )  # 前向传播层
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)  # 多头注意力

        x = self.dropout(self.norm1(attention + query))  # 先残差结果再层归一化在dropout
        forward = self.feed_forward(x)  # 前向传播
        out = self.dropout(self.norm2(forward + x))  # 先残差结果再层归一化在dropout
        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,  # 源语言字典大小
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
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # 词嵌入
        self.position_embedding = nn.Embedding(max_length, embed_size)  # 位置嵌入

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)]
        )  # 多层编码器层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_lengh = x.shape  # 批次大小和句子长度
        positions = torch.arange(0, seq_lengh).expand(
            N, seq_lengh).to(self.device)  # 生成位置向量
        out = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))  # 词嵌入和位置嵌入相加

        for layer in self.layers:
            out = layer(out, out, out, mask)  # 多层编码器层

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)  # 多层编码器层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)  # 掩码注意力层
        query = self.dropout(self.norm(attention + x))  # 先残差结果再层归一化在dropout
        out = self.transformer_block(value, key, query, src_mask)  # 多层编码器层
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,  # 目标语言字典大小
                 embed_size,  # 词向量维度
                 num_layers,  # 解码器层数
                 heads,  # 多头注意力头数
                 forward_expansion,  # 前向传播扩展倍数
                 dropout,
                 device,
                 max_length):  # 最大长度
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)  # 词嵌入
        self.position_embedding = nn.Embedding(max_length, embed_size)  # 位置嵌入

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )  # 多层解码器层
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)  # 输出全连接层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape  # 批次大小和句子长度
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)  # 生成位置向量
        x = self.dropout((self.word_embedding(
            x) + self.position_embedding(positions)))  # 词嵌入和位置嵌入相加

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,  # 源语言字典大小
                 trg_vocab_size,  # 目标语言字典大小
                 src_pad_idx,  # 源语言填充标记
                 trg_pad_idx,  # 目标语言填充标记
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))  # 1111
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 11111

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask,  trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [
                       1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size,
                        src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
