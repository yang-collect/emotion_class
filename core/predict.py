import os
import sys

import torch
import time
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config


def predict(data, model, tokenizer):
    """
    对传入的一条数据进行tokenizer，并将结果传递给模型，让模型预测结果
    """
    # 对传入的一条数据进行tokenizer
    token = tokenizer(data,
                      add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                      max_length=config.max_length,  # 设定最大文本长度
                      padding=True,  # padding操作
                      truncation=True,  # truncation
                      is_split_into_words=True,  # 是否分词
                      return_tensors='pt'  # 返回的类型为pytorch tensor
                      )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # print(token)
    # 模型eval过程
    model.eval()
    with torch.no_grad():
        # 将tokenizer后的数据转换为一个字典
        data = {k: v.to(device) for k, v in token.items()}
        # print(data)
        # 以字典解码的方式传入数据给模型
        out = model(**data)
        # print(out.logits)
        # 对模型输出logits进行softmax归一化，得到预测概率
        prob = F.softmax(out.logits, dim=-1)
        # 取预测概率中最大的下标，这里刚好为对应的下标0，1，2即为标签
        prediction = torch.argmax(prob, dim=-1)

    return prob, prediction


if __name__ == '__main__':
    model_path = config.save_path
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    # tokenizer实例化
    tokenizer = BertTokenizer.from_pretrained(model_path)
    start=time.time()
    text = '靠 你 真是 说 废话 '.split()
    probs, preds = predict(text, model, tokenizer)
    print(probs.numpy(), preds.numpy().tolist()[0])
    print(time.time()-start)
