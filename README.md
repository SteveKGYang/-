#阅读理解问答任务

##预测方法

将要预测的文件使用preprocess.py处理后在predict代码中输入相应数据集地址，之后将得到的文件放入validate文件夹中计算F1值。


##相关文件

###data

包括用于训练的数据，其中divided data用于finetune之后的模型的训练。我们将preprocessed后的数据分为2800条训练数据和1000条开发数据进行final\_model\_split的训练,并且使用未分的数据用于final\_model\_whole的训练。

###cmrc-2018-master

这是第二届”讯飞杯“中文机器阅读理解评测的数据集，我们使用该数据集进行finetune。

###pretrained\_bert

这是下载的bert预训练模型，使用全词mask进行训练。下载网址：https://github.com/ymcui/Chinese-BERT-wwm

###finetune\_model

这是使用第二节讯飞杯机器阅读理解数据finetune之后的预训练模型。

###final\_model\_split

使用切分数据train.text的训练模型

###final\_model\_whole

使用未切分数据whole\_train.text训练的模型

###finetune\_data

使用finetune\_data\_preprocess.py处理后的数据，用于对bert模型进行finetune

###finetune\_bert.py

该代码用于训练finetune后的bert模型使其适用于该数据集。

###finetune\_data\_preprocess.py

对讯飞数据进行预处理的代码。

###preprocess.py

对训练数据进行预处理，包括对过长数据的处理。

###train\_bert.py

对finetune好的模型进行训练使其适用于当前数据集。

###predict.py

加载训练好的模型并将测试数据输入得到预测结果。


##相关文档

pytorch版本BertForQuestionAnswering的文档：https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforquestionanswering#bertforquestionanswering

huggingface项目地址：https://github.com/huggingface/transformers

##相关结果

使用divided\_data中的数据得到的验证集F1值为：71.280646

使用测试数据得到F1值为70.0119
