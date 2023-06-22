# ConTKGQA
We provide the top-performing model: Model 4, as described in the paper, which combines BERT with memory networks.

You can find the detailed log file of our testing in the following location: `experiments/logs/test_mem_detailed.log`. This log file contains all information about our experiment results.

## Dataset
Please download the dataset [CronQuestions](https://github.com/apoorvumang/CronKGQA) and save it under the folder "datasets/CronQuestions"

The trained model exceeds the upload limit. Please reach out to the author directly to obtain the model.
## Requirements
Python version >= 3.9

Pytorch >= 1.10.0
```
pip install -r requirements.txt
```

## Train

```
python train.py --valfreq 5
```
`valfreq` refers to the evaluation frequency during the training process. If its value is set to 5, it indicates that the model will be evaluated after every 5 epochs.

## Test

```
python test.py --checkpoint CHECKPOINTNAME
```

