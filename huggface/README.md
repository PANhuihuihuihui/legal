# leagal bert pretraining in HKLII


## Table of content
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Experiment result](#experiment)
- [License](#license)
## Backgroud
- HKLII dataset stat
|:----:|:--------:|
|count |  2.851737e+06 |
|mean  |   8.366615e+01|
|std   |   7.347763e+01|
|min   |   8.000000e+00|
|25%   |   3.400000e+01|
|50%   |   6.500000e+01|
|75%   |   1.120000e+02|
|max   |   1.181900e+04|
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

prtraining material are all from HKLII

| Model name        |  Method   | Perplexity (before)   | Perplexity (after)| Vocb size |
|:-----------       |:--------: |:------------------:   |:-----------------:|:---------:|
| custom-legalbert  | mask      | 18.370377004          | 4.567             | 32000     |
| legal-bert-base   | mask      | 11.055                | 3.376             | 30522     |
| multilingual bert | mask      | 15.6468               | 3.632             | todo      |
| custom-legalbert  | mask+cls  | todo                  |todo               | todo      |
| legal-bert-base   | mask+cls  | todo                  |todo               | todo      |
| multilingual bert | mask+cls  | todo                  |todo               | todo      |

because of 

### finetune on downstream task

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



