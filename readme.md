# 数据介绍

数据来源于[这里](https://aistudio.baidu.com/aistudio/datasetdetail/12605/0)，是百度飞桨平台上的一份开源数据集，是一个三分类任务：其中0表示消极；1表示中性；2表示积极。

# 文件结构介绍

```sh
│  readme.md
│
├─.idea
│
├─core  # 模型的运行、评估、部署文件
│      emotion_identify_sever.py # flask部署文件
│      eval.py # 计算accuracy
│      predict.py # 单条数据预测样例
│      train.py # 训练
│
├─data # 数据文件
│      dev.txt
│      infer.txt
│      test.txt
│      train.txt
│      vocab.txt
│
├─lib # 数据处理和config文件
│  │  config.py
│  │  DataProcess.py
│  │  __init__.py
│  │
│  └─__pycache__
│          config.cpython-37.pyc
│          DataProcess.cpython-37.pyc
│          __init__.cpython-37.pyc
│
└─model_file # 模型文件
    │  
    ├─emotion_identify # fine-tune的模型，模型文件过大不方便上传
    │      config.json
    │      pytorch_model.bin
    │      special_tokens_map.json
    │      tokenizer.json
    │      tokenizer_config.json
    │      vocab.txt
    │
    └─ernie-1.0 # 下载的预训练模型，下载链接https://huggingface.co/nghuyong/ernie-1.0/tree/main
            config.json
            pytorch_model.bin
            special_tokens_map.json
            tokenizer_config.json
            vocab.txt

```

# 模型结果

`core/eval.py`输出结果：在训练和开发集上都有着99.4%左右的准确率，在测试集上也有90.8%的准确率

![image-20210824164903870](https://github.com/yang-collect/emotion_class/blob/main/image-20210824164903870.png)



`core/emotion_identify_sever.py`用postman测试结果： 用时76ms

![image-20210824163800123](https://github.com/yang-collect/emotion_class/blob/main/image-20210824163800123.png)

# 主要用到的模型、环境以及流程介绍

下文实现主要使用pytorch以及hugging face开源的transformers

我训练环境为显卡为3080，测试环境是r7 5800h 核显笔记本

> torch                     1.8.1+cpu                pypi_0    pypi

> pytorch-transformers      1.2.0                    pypi_0    pypi

> transformers              4.6.1                    pypi_0    pypi

## 主要用到模型是百度开源的ernie-1.0

> 关于选择模型的原因由很多考虑：
>
> - 首先，传统的基于词向量化的ligtgbm、textcnn等，极度依赖词向量的学习，词向量的学习和构建是一个十分庞大且需要计算资源的过程；虽然由开源的词向量，且开源词向量通常是来自很多开源语料，但开源词向量并未考虑词在当前语料中的情况和语序的问题；
> - 其次，基于序列的rnn、lstm、gru虽然在一定程度上解决了词向量的表征和语序的问题，但同时引入了一新的问题就是计算无法并行处理，由于rnn等一系列模型的当前状态的计算是依赖于上一个状态的计算，其本身的设计决定这一系列模型是串行的；而且单向的rnn计算中当前词的计算包含之前所有词的信息，从一定程度上也加深了模型的噪声。并且缺失了后文的信息，后来引入的attention 机制和elmo的双向状态计算可以一定程度上解决前面的两个问题；
> - 再者，bert模型来自于transformer结构的编码器部分，对词向量的语序部分采用相对位置编码来刻画，对计算的并行部分采用多头注意力机制来并行计算，对上下文部分采用随机掩码来预测被掩码部分的自学习方式来得到当前词的上下文；
> - 最后，虽然bert模型由很不错的效果，并且开辟了预训练时代，但是原始bert的中文模型是基于字粒度的模型，中文的语境和语义较为复杂，单独用字模型可能无法很好表征其含义；百度的ernie-1.0 是基于中文网站语料，且考虑了数据中的词、实体以及实体关系，来构建的模型，相比于原始bert更能表征出语言本身的含义。

## 流程

1. 数据加载至dataloder

   > 先将数据读取并实现自己的dataset类，在dataset类中规定好数据输出的格式，再将数据加载至dataloder
   >
   > ```python
   > class MyDataset(Dataset):
   >        """ 自定义dataset，实现dataset的基本功能，传递给DataLoader
   >        """
   > 
   >        def __init__(self, data, tokenizer):
   >            """
   >            :param data: 传入数据
   >            :param tokenizer: tokenizer获取元素的input_ids和
   >            """
   >            self.tokenizer = tokenizer
   >            if len(data) == 2:
   >                self.text, self.label = data
   >                self.length = len(self.text)
   >                self.input_ids, self.mask = self.encode(self.text)
   >                # 将label数据转换为tensor
   >                self.label = torch.tensor(self.label)
   >            else:
   >                # 需要predict的数据
   >                self.text, self.mask = self.encode(list(data))
   >    
   >        def encode(self, text_list):
   >            token = self.tokenizer(
   >                text_list,
   >                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
   >             max_length=config.max_length,  # 设定最大文本长度
   >                padding=True,  # padding
   >                truncation=True,  # truncation
   >                is_split_into_words=True,  # 是否分词
   >                return_tensors='pt'  # 返回的类型为pytorch tensor
   >            )
   >            return token['input_ids'], token['attention_mask']
   >    
   >        def __getitem__(self, index):
   >            if self.label is not None:
   >                return {'input_ids': self.input_ids[index], 'attention_mask': self.mask[index], 'labels': self.label[index]}
   >            else:
   >                return {'input_ids': self.input_ids[index], 'attention_mask': self.mask[index]}
   >    
   >        def __len__(self):
   >            return self.length
   >    ```
   >    
   >    
   
2. 将dataloder中的数据按照批次循环epochs次，每个批次构建一个字典传递给预训练模型

   > 传递给预训练模型需要使用字典的解码操作，需要在dataset的构造上与接受参数保持一致，具体构建dataset可参考上述代码中`def __getitem__(self, index)`

3. 模型训练以及梯度更新

   > 构建一个测试样本在无梯度条件下的evaluation函数，在训练的同时输出训练集和测试的损失或者其他评价指标，保存模型

4. 模型预测

   > 对输出的文本获取其情感倾向

5. 模型部署

   > 模型部署选择flask进行部署

