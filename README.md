# RefNet (AAAI 2020 oral paper)
The code for [
RefNet: A Reference-aware Network for Background Based Conversation](https://arxiv.org/abs/1908.06449)


## Requirements 
python 3.6
tensorflow-gpu 1.3

## Run training
To train your model, run:

```
python run.py --mode=train
```

This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

**Increasing sequence length during training**: Note that to obtain the results described in the paper, we increase the values of `max_enc_steps` and `max_dec_steps` in stages throughout training (mostly so we can perform quicker iterations during early stages of training). If you wish to do the same, start with small values of `max_enc_steps` and `max_dec_steps`, then interrupt and restart the job with larger values when you want to increase them.

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
