from flask import Flask, request, Response, jsonify
from concurrent.futures import ThreadPoolExecutor
from transformers import BertForSequenceClassification, BertTokenizer
import json
import datetime, time
import torch
import torch.nn.functional as F

import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config


class MyResponse(Response):
    @classmethod
    def force_type(cls, response, environ=None):
        if isinstance(response, (list, dict)):
            response = jsonify(response)
        return super(Response, cls).force_type(response, environ)


# 创建服务
server = Flask(__name__)
# 约定flask接受参数的类型
server.response_class = MyResponse
# 创建一个线程池，默认线程数量为cpu核数的5倍
executor = ThreadPoolExecutor()
# fine-tune模型路径
model_path = config.save_path
# 加载模型
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
# tokenizer实例化
tokenizer = BertTokenizer.from_pretrained(model_path)

index2label = {0: '消极', 1: '积极', 2: '中性'}


# 绑定目录以及方法
@server.route('/text_classification/emotion_identify', methods=["POST"])
def scene_object_appearance_class():
    data = request.get_json()
    output_res = {}
    if len(data) == 0:
        output_res["status"] = "400"
        output_res["msg"] = "Flase"
        output_res['text_label'] = "Flase"
        return output_res
    else:
        try:
            # 对传入的一条数据进行tokenizer
            token = tokenizer(data['text'].split(),
                              add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                              max_length=config.max_length,  # 设定最大文本长度
                              padding=True,  # padding
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

            # 获取label对应的文本
            output_res['label'] = index2label[prediction.numpy().tolist()[0]]
            return json.dumps(output_res, ensure_ascii=False)
        except Exception as e:
            print("异常原因: ", e)
            return {"error": 500}


def host():
    """ main 函数
    """
    HOST = '0.0.0.0'
    # 服务端口，为外部访问
    PORT = 5019
    server.config["JSON_AS_ASCII"] = False
    server.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    nowTime1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime1)

    host()

    nowTime2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime2)
