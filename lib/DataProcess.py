import torch
from torch.utils.data import Dataset, DataLoader
# import os
# import sys
#
# abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(abs_path)

import config


def read_data(path=config.train_path):
    r""" 读取json格式的数据
    :param path: 文件路径
    :return: 若数据位train则返回，dict形式的数据，否则返回hash 表形式的数据
    """

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        text, label = [], []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.split('\t')
            label.append(int(line[0]))
            text.append(line[1].split())
    return text, label


class MyDataset(Dataset):
    """ 自定义dataset，实现dataset的基本功能，传递给DataLoader
    """

    def __init__(self, data, tokenizer):
        """
        :param data: 传入数据
        :param tokenizer: tokenizer获取元素的input_ids和
        """
        self.tokenizer = tokenizer
        if len(data) == 2:
            self.text, self.label = data
            self.length = len(self.text)
            self.input_ids, self.mask = self.encode(self.text)
            # 将label数据转换为tensor
            self.label = torch.tensor(self.label)
        else:
            # 需要predict的数据
            self.text, self.mask = self.encode(list(data))

    def encode(self, text_list):
        token = self.tokenizer(
            text_list,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=config.max_length,  # 设定最大文本长度
            padding=True,  # padding
            truncation=True,  # truncation
            is_split_into_words=True,  # 是否分词
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        return token['input_ids'], token['attention_mask']

    def __getitem__(self, index):
        if self.label is not None:
            return {'input_ids': self.input_ids[index], 'attention_mask': self.mask[index], 'labels': self.label[index]}
        else:
            return {'input_ids': self.input_ids[index], 'attention_mask': self.mask[index]}

    def __len__(self):
        return self.length


def DataLoad(tokenizer, train_path=config.train_path, test_path=config.test_path):
    # 加载训练和测试数据
    train_data = read_data(train_path)
    test_data = read_data(test_path)

    # 将训练和测试数据tokenizer并实例化为dataset类
    train_dataset = MyDataset(train_data, tokenizer)
    test_dataset = MyDataset(test_data, tokenizer)

    # 将上面实例化的dataset传递给dataloader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    test_dataloder = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    return train_dataloader, test_dataloder


# if __name__ == '__main__':
#     # text, label = read_data()
#     # print(text, label)
#     from transformers import AutoTokenizer, AutoModel
#
#     model = AutoModel.from_pretrained(config.model_path, num_labels=3)
#     # tokenizer实例化
#     tokenizer = AutoTokenizer.from_pretrained(config.model_path)
#     train_dataloader, test_dataloder = DataLoad(tokenizer)
#     print(next(iter(train_dataloader)))
#     print('=' * 50)
#     print(next(iter(test_dataloder)))
