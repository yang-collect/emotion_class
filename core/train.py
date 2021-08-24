import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_scheduler
import numpy as np

import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config, DataProcess


def train(train_path=config.train_path, test_path=config.test_path,
          epochs=config.epochs, model_path=config.model_path):
    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    # tokenizer实例化
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 实例化优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 加载数据
    train_dataloader, test_dataloder = DataProcess.DataLoad(tokenizer, train_path, test_path)

    num_training_steps = epochs * len(train_dataloader)
    # warm up
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # 模型的训练过程
    model.train()
    # 初始化验证集损失
    val_loss = np.inf
    for epoch in range(epochs):
        batch_loss = []
        for num, batch in enumerate(train_dataloader):
            # 以字典解码形式加载数据到模型中
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # 将模型的loss记录下来
            loss = outputs.loss
            batch_loss.append(loss.item())
            # 梯度更新
            loss.backward()
            # 优化器和学习率更新
            optimizer.step()
            lr_scheduler.step()
            # 梯度清零
            optimizer.zero_grad()
            # 每100个打印一次结果
            if num % 100 == 0:
                print(f'epoch:{epoch},batch :{num} ,train_loss :{loss} !')

        epoch_loss = np.mean(batch_loss)
        avg_val_loss = compute_loss(model, test_dataloder)
        print(f'epoch:{epoch},tran_loss:{epoch_loss},valid loss;{avg_val_loss}')
        print('*' * 100)
        # Update minimum evaluating loss.
        if (avg_val_loss < val_loss):
            tokenizer.save_pretrained(config.save_path)
            model.save_pretrained(config.save_path)
            val_loss = avg_val_loss

    print(val_loss)


def compute_loss(model, val_data):
    """Evaluate the loss for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')

    val_loss = []
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            data = {k: v.to(device) for k, v in data.items()}
            out = model(**data)
            loss = out.loss
        val_loss.append(loss.item())

    return np.mean(val_loss)


if __name__ == '__main__':
    train()
