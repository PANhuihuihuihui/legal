# leagal bert pretraining in HKLII


## Table of content
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Usage
1. install the environment with conda
2. config the model and run
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
## Experment result

### pertraining
> prtraining material are all from HKLII

| Model name        |  Method   | Perplexity (before)   | Perplexity (after)| Vocb size |
|:-----------       |:--------  |:------------------:   |:-----------------:|:----------|
| custom-legalbert  | mask      | 53254.53526           | 4.567             | 32000     |
| legal-bert-base   | mask      | 11.055                | 3.376             | 30522     |
| multilingual bert | mask      | todo                  |todo               | todo      |
| custom-legalbert  | mask+cls  | todo                  |todo               | todo      |
| legal-bert-base   | mask+cls  | todo                  |todo               | todo      |
| multilingual bert | mask+cls  | todo                  |todo               | todo      |

### finetune

| Model name        |  Method   | HKLII   | Perplexity (after)| Vocb size |
|:-----------       |:--------  |:------------------:   |:-----------------:|:----------|
| custom-legalbert  | mask      | 53254.53526           | 4.567             | 32000     |
| legal-bert-base   | mask      | 11.055                | 3.376             | 30522     |
| multilingual bert | mask      | 
| custom-legalbert  | mask+cls  |
| legal-bert-base   | mask+cls  |
| multilingual bert | mask+cls  |






