from datasets import load_metric
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config, DataProcess


def DataLoad(tokenizer, dev_path=config.dev_path):
    """
    加载开发数据集以及将其转换为dataloder
    """
    # 数据读取
    dev_data = DataProcess.read_data(dev_path)
    # 构建dataset类
    dev_dataset = DataProcess.MyDataset(dev_data, tokenizer)
    # 构建dataloder类
    dev_dataloder = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size)

    return dev_dataloder


def evaluate(model, eval_dataloader):
    """
    根据传入模型以及数据集计算accuracy
    """
    # 加载accuracy评估器
    metric = load_metric("accuracy")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型评估过程
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # 获取模型输出
        logits = outputs.logits
        # 对模型输出取argmax
        predictions = torch.argmax(logits, dim=-1)
        # 将当前批次数据的预测结果和原始结果传递给评估器
        metric.add_batch(predictions=predictions, references=batch["labels"])
    # 返回评估其计算结果
    return metric.compute()


if __name__ == '__main__':
    # 模型保存路径
    model_path = config.save_path
    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    # tokenizer实例化
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 训练即测试数据加载
    train_dataloder, test_dataloder = DataProcess.DataLoad(tokenizer)
    # 打印训练的accuracy
    print('train data:', evaluate(model, train_dataloder))
    # 打印测试数据上的accuracy
    print('test data:', evaluate(model, test_dataloder))
    # 加载评估数据
    eval_dataloader = DataLoad(tokenizer)
    # 打印在评估数据上的accuracy
    print('dev data', evaluate(model, eval_dataloader))
