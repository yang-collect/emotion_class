from pathlib import Path
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

project_path = Path(__file__).parent.parent
train_path = project_path.joinpath('./data/train.txt')
test_path = project_path.joinpath('./data/test.txt')
dev_path = project_path.joinpath('./data/dev.txt')
infer_path=project_path.joinpath('./data/infer.txt')

# 模型参数
model_path = project_path.joinpath('./model_file/ernie-1.0') # 预训练模型路径
# project_path.joinpath('./model_file/ernie-1.0')
epochs = 6
max_length = 128
batch_size = 32
# print(train_path)
save_path = project_path.joinpath('./model_file/emotion_identify')
# print(save_path)

