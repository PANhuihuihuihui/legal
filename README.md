# leagal bert pretraining in HKLII


## Table of content
- [ToDo](#todo)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Experiment result](#experiment)
- [License](#license)

## TODO
@ 3.25
- [x] 与 Xiangwei  对接pretrain的下游任务
- [ ] 训练代码优化
    - [x] cls 的训练代码： 查看标准，实现和训练
    - [x] 加入法研杯的训练数据,new mbert.
    - [ ] Roberta 训练策略
    - [ ] 梯度累计 增大batch size
- [ ] 继续整理Ben那边的任务, 标注到下游任务

## Backgroud
- HKLII dataset stat(summary of token length of in each paragraph)

    | stat value | value    |
    |:----------:|:--------:|
    |count |  2.851737e+06  |
    |mean  |   83.66615     |
    |std   |   73.47763     |
    |min   |   8.0          |
    |25%   |   34           |
    |50%   |   65           |
    |75%   |   112          |
    |max   |   11819        |

- HKLII + caill

    | stat value | value    |
    |:----------:|:--------:|
    |count |  2.883591e+06  |
    |mean  |   87.13742     |
    |std   |   949.2628     |
    |min   |   8.0          |
    |25%   |   34           |
    |50%   |   66           |
    |75%   |   114          |
    |max   |   17415        |

- [Datasets format](data_prepare/Readme.md)
    

## Install 
```sh
cd huggface
conda env create -f environment.yml
```
## Usage

config the model and run
- local 
    ```sh
    python train.py
    ```
- distributed mode
    ```sh
    CUDA_VISIBLE_DEVICES={your devices} python -m torch.distributed.launch  --nproc_per_node={\# your divce} train.py
    ```
    eg:
    ```sh
    CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch  --nproc_per_node=3 train.py
    ```
## Experiment

### pertraining

| Model name                |  Method   | Perplexity (before)   | Perplexity (after)| Vocb size |
|:----------------------    |:--------: |:------------------:   |:-----------------:|:---------:|
| [custom-legalbert][1]     | mask      | 18.370377004          | 4.567             | 32000     |
| [legal-bert-base][2]      | mask      | 11.055                | 3.376             | 30522     |
| [multilingual bert][3]    | mask      | 15.6468               | 2.850             | 105879    |
| [custom-legalbert][1]     | mask+cls  | todo                  |todo               | todo      |
| [legal-bert-base][2]      | mask+cls  | todo                  |todo               | todo      |
| [multilingual bert][3]    | mask+cls  | todo                  |todo               | todo      |

because in HKLII datasets it is possible to have chinese and english in same document multilingual 

### Finetune on downstream task

1. [HKLII](https://www.hklii.hk/eng/)

    | Model name        |  Method   |  Task  | Perplexity (after)| Vocb size |
    |:-----------:      |:--------  |:------------------:   |:-----------------:|:---------:|
    | custom-legalbert  | mask      | todo     | todo       | todo     |
    | legal-bert-base   | mask      | todo     | todo       | todo      |
    | multilingual bert | mask      | 
    | custom-legalbert  | mask+cls  |
    | legal-bert-base   | mask+cls  |
    | multilingual bert | mask+cls  |

2. [法研杯]()

    | Model name        |  Method   |  Task  | Perplexity (after)| Vocb size |
    |:-----------:      |:--------  |:------------------:   |:-----------------:|:---------:|
    | custom-legalbert  | mask      | training     | training       | training     |
    | legal-bert-base   | mask      | training     | training       | training      |
    | multilingual bert | mask      | 
    | custom-legalbert  | mask+cls  |
    | legal-bert-base   | mask+cls  |
    | multilingual bert | mask+cls  |



## License

[MIT](LICENSE) © Huijie Pan 


[1]:https://huggingface.co/zlucia/custom-legalbert
[2]:https://huggingface.co/nlpaueb/legal-bert-small-uncased
[3]:bert-base-multilingual-uncased


