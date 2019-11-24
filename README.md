# RefNet (AAAI 2020 oral paper)
The code for [
RefNet: A Reference-aware Network for Background Based Conversation](https://arxiv.org/abs/1908.06449)

![image](https://github.com/ChuanMeng/RefNet/blob/master/model.jpg)

If you use any source code included in this repo in your work, please cite the following paper.
```
@inproceedings{chuanmeng2020refnet,
 author = {Chuan Meng, Pengjie Ren, Zhumin Chen, Christof Monz, Jun Ma, Maarten de Rijke},
 booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
 title = {RefNet: A Reference-aware Network for Background Based Conversation},
 year = {2020}
}
```
## Requirements 
*python 3.6

*tensorflow-gpu 1.3

## Run dataset preprocessing
First, you should download the [raw data version of Holl-E](https://github.com/nikitacs16/Holl-E), and put the raw data files (train_data.json, dev_data.json and test_data.json）in the directory `/data`.
Then run the preprocessing script in the directory `/data`:
```
python preprocress.py
```
This will create the data in the setting of `mixed-short` background, which is the data version used in our paper. You can also change the setting in `preprocress.py` to create the data in the setting of `oracle` or `mixed-long` background.

## Run training
To train your model, run:

```
python run.py --mode=train
```

This will create a directory `log/RefNet/train/` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Hyperparameter**: If you want to change some hyperparameters, you can change some settings in the file `config.yaml` carefully. 

## Run validation
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the results of automatic evaluation (F1, BLEU-4, ROUGE-1, ROUGE-2 and ROUGE-L ). To do this, run:

```
python run.py --mode=val
```
This will create `val_result.json` and `Validation_Infer_ckpt-xxxx` in the directory `log/RefNet/`, where `val_result.json` records the model performance about automatic evaluation metrics on the validation set for every epoch, and `Validation_Infer_ckpt-xxxx` records the response outputted by our model.

**Note**: you can run validation after finishing the training, or at the same time with training. 


## Run testing
After finishing the proprocess of training and validation, you should open the file `val_result.json` to select the best model based on the validation results according to BLEU metric or other metrics you like. Next, you should copy the model file path, for example`log/RefNet/train/model.ckpt-10775`, and run testing on this model:

```
python run.py --mode=test --appoint_test log/RefNet/train/model.ckpt-10775
```
This will create `test_result.json` and `Test_Infer_ckpt-10775` in the directory `log/RefNet/`, where the purpose of these two files is consistent with the validation process.


