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
 year = {2019}
}
```
## Run dataset preprocessing
First, you should download the [raw data version of Holl-E](https://github.com/nikitacs16/Holl-E), and put the raw data files (train_data.json, dev_data.json and test_data.json）in the directory `/data`.
Then run the preprocessing script in the directory `/data`:
```
python preprocress.py
```
This will create the data in the setting of `mixed-short` background, which is the data version used in our paper. You can also change the setting in `preprocress.py` to create the data in the seting of `oracle` and `mixed-long` background.

## Requirements 
python 3.6

tensorflow-gpu 1.3

## Run training
To train your model, run:

```
python run.py --mode=train
```

This will create a subdirectory of your specified `log` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**hyperparameter**: If you want to change some hyperparameters, you can change some settings in the file `config.yaml` carefully. 


## Run validation
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:

```
python run.py --mode=val
```

Note: you want to run the above command using the same settings you entered for your training job.

**Restoring snapshots**: The eval job saves a snapshot of the model that scored the lowest loss on the validation data so far. You may want to restore one of these "best models", e.g. if your training job has overfit, or if the training checkpoint has become corrupted by NaN values. To do this, run your train command plus the `--restore_best_model=1` flag. This will copy the best model in the eval directory to the train directory. Then run the usual train command again.

## Run testing
To run beam search decoding:

```
python run.py --mode=test --appoint_test log/RefNet/train/model.ckpt-10775
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.


