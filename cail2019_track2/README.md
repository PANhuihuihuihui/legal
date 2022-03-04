欢迎大家使用[tensorflow1.x的bert系列模型库，支持单机多卡，梯度累积，自动导出pb部署](https://github.com/huanghuidmml/textToy)
**代码说明**
-------

#### **基本代码**
CUDA_VISIBLE_DEVICES=1是指定第一块显卡，根据具体情况自己改，
如果CPU的话就不用了。

**训练**

> CUDA_VISIBLE_DEVICES=1 python train.py

**将ckpt转为pb**

> CUDA_VISIBLE_DEVICES=1 python convert.py

**线下测试**

> CUDA_VISIBLE_DEVICES=1 python evaluation.py

#### **如果需要额外预训练的话，使用以下代码**

**创建预训练数据txt**

> python genPretrainData.py

**创建预训练数据的tfrecord文件**

> python createPretrainData.py

**预训练**

> CUDA_VISIBLE_DEVICES=1 python run_pretrain.py

**Reference**
-----
1. [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
2. [The implementation of focal loss proposed on "Focal Loss for Dense Object Detection" by KM He and support for multi-label dataset.](https://github.com/ailias/Focal-Loss-implement-on-Tensorflow)

**感谢**
-----
感谢队友牧笛的帮助
