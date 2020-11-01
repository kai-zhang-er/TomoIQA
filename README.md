# TomoIQA
Image quality assessment for tomography images. The corresponding documents can be found in [TomoIQA_report](https://summerof15.github.io/publications/assets/files/TomoIQA_report.pdf) and [TomoIQA_slides](https://summerof15.github.io/publications/assets/files/TomoIQA_slides.pdf)

## Folder structure
```
.				# project use the tensorflow 2.x
├── datset
│   ├── origin			# Origin images
│   ├── distorted		# generated distorted images
│   ├── config			# txt files for training and testing
│   ├── testset			# directory for prediction
├── tf1				# project use the tensorflow 1.x
├── metrics			# used for label projection
└── ...	
```
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow.

```bash
pip install tensorflow, sacred
```

## Usage
Only the sample dataset is currenntly provided in this project. The whole dataset will be provided in the future.

### Dataset preparation

1. Modify the `config.json` file to customize the configuration of the project
2. put the original image in `dataset/origin/` and generate the distorted images
```bash
python add_distortion.py
```
3. generate the `train_finetune.txt` for training the regression procedure

```python
python generate_labels.py
```

### Training

1. train the rank process

```bash
python train_rank_v2.py
```

2. train the finetune process

```
python tran_finetune.py
```

### Prediction

Modify the `test_finetune.txt`  and set the image name

```bash
python pred_finetune.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
