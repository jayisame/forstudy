"""
数据预处理
"""
import os
import re
from nltk.corpus import stopwords
import nltk
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 下载停用词
# nltk.download('stopwords')
# english_stops = set(stopwords.words('english'))


def data_get(flag):
    """读取数据
    """
    if flag == 'train':
        file_path = "C:\\Users\\hcjtx\\PycharmProjects\\transfomer实验\\train"
    else:
        file_path = "C:\\Users\\hcjtx\\PycharmProjects\\transfomer实验\\test"

    # 读取文本文件,构建文本列表
    data_list = []  # 存储文本和标签的元组
    positive_dir = os.path.join(file_path, 'pos')
    negative_dir = os.path.join(file_path, 'neg')

    # 加载积极评价（标签 1）
    for filename in os.listdir(positive_dir):
        filepath = os.path.join(positive_dir, filename)
        if filename.endswith(".txt"):  # 确保是文本文件
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                # 去掉html标签
                text = re.sub(r'<.*?>', '', text)
                # 删除标点和数字
                text = re.sub(r'[^A-Za-z]', ' ', text)
                # 分词
                words = [w for w in text.split()]
                # 将大写字母转成小写
                words = [w.lower() for w in words]
                data_list.append((words, 1))

    # 加载消极评价（标签 0）
    for filename in os.listdir(negative_dir):
        filepath = os.path.join(negative_dir, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                # 去掉html标签
                text = re.sub(r'<.*?>', '', text)
                # 删除标点和数字
                text = re.sub(r'[^A-Za-z]', ' ', text)
                # 分词并过滤掉停用词
                words = [w for w in text.split()]
                # 将大写字母转成小写
                words = [w.lower() for w in words]
                data_list.append((words, 0))

    return data_list


def build_dict(corpus):
    """构建词典

    Args:
        corpus (_type_): data_get函数返回的列表

    Returns:
        _type_:转换词典
    """
    word_freq_dict = dict()
    # 统计词频
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1
    # 按词频降序排序
    word_freq_dict = sorted(word_freq_dict.items(),
                            key=lambda x: x[1], reverse=True)

    word2id_dict = dict()

    # 未登录词记作[oov]，填充词记作[pad]
    word2id_dict['[oov]'] = 0
    word2id_dict['[pad]'] = 1

    for word, _ in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)

    return word2id_dict


def convert_corpus_to_id(corpus, word2iddict):
    """映射句子和标签到id

    Args:
        corpus (_type_): 词语标签列表
        word2id_dict (_type_): 转换词典

    Returns:
        _type_: id标签列表
    """
    data_set = []
    for sentence, sentence_label in corpus:
        sentence = [word2iddict[word] if word in word2iddict
                    else word2iddict['[oov]'] for word in sentence]

        data_set.append((sentence, sentence_label))
    return data_set


class IMDBDataset(Dataset):
    """自定义数据类

    Args:
        Dataset (_type_): 继承Dataset类,适应PyTorch数据加载器
    """

    def __init__(self, data):
        super(IMDBDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        return sentence, label


def collate_fn(batch):
    sentences, labels = zip(*batch)
    # 动态填充到批次中最长句子的长度,并转为张量
    padded_sentences = pad_sequence([torch.LongTensor(
        s) for s in sentences], batch_first=True, padding_value=1)
    labels = torch.LongTensor(labels)
    return padded_sentences, labels
