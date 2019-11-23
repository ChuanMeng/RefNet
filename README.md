## Baseline (GTTP) for Towards Exploiting Background Knowledge for Building Conversation Systems (EMNLP 2018)
This code is modified [Pointer Generator](https://github.com/abisee/pointer-generator) code. It uses Tensorflow 1.0

We add the query decoder which was not present in the original architecture. Please refer to the Appendix for additional details.

## How to run (Instructions as per the original Repository)

### Run training
To train your model, run:

```
python run_summarization.py --mode=train --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

**Increasing sequence length during training**: Note that to obtain the results described in the paper, we increase the values of `max_enc_steps` and `max_dec_steps` in stages throughout training (mostly so we can perform quicker iterations during early stages of training). If you wish to do the same, start with small values of `max_enc_steps` and `max_dec_steps`, then interrupt and restart the job with larger values when you want to increase them.

### Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:

```
python run_summarization.py --mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job.

**Restoring snapshots**: The eval job saves a snapshot of the model that scored the lowest loss on the validation data so far. You may want to restore one of these "best models", e.g. if your training job has overfit, or if the training checkpoint has become corrupted by NaN values. To do this, run your train command plus the `--restore_best_model=1` flag. This will copy the best model in the eval directory to the train directory. Then run the usual train command again.

### Run beam search decoding
To run beam search decoding:

```
python run_summarization.py --mode=decode --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

**Visualize your output**: Additionally, the decode job produces a file called `attn_vis_data.json`. This file provides the data necessary for an in-browser visualization tool that allows you to view the attention distributions projected onto the text. To use the visualizer, follow the instructions [here](https://github.com/abisee/attn_vis).

If you want to run evaluation on the entire validation or test set and get ROUGE scores, set the flag `single_pass=1`. This will go through the entire dataset in order, writing the generated summaries to file, and then run evaluation using [pyrouge](https://pypi.python.org/pypi/pyrouge). (Note this will *not* produce the `attn_vis_data.json` files for the attention visualizer).

## How to run (Simplified)
We use the following command to run the train and validation simultaneously:
```
python automate.py config.yaml
```
Note you can have different file location of config.yaml

In order to test, you can follow the same commands as the original repository.
### Evaluate with ROUGE
`decode.py` uses the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. `pyrouge` provides an easier-to-use interface for the official Perl ROUGE package, which you must install for `pyrouge` to work. Here are some useful instructions on how to do this:
* [How to setup Perl ROUGE](http://kavita-ganesan.com/rouge-howto)
* [More details about plugins for Perl ROUGE](http://www.summarizerman.com/post/42675198985/figuring-out-rouge)

**Note:** As of 18th May 2017 the [website](http://berouge.com/) for the official Perl package appears to be down. Unfortunately you need to download a directory called `ROUGE-1.5.5` from there. As an alternative, it seems that you can get that directory from [here](https://github.com/andersjo/pyrouge) (however, the version of `pyrouge` in that repo appears to be outdated, so best to install `pyrouge` from the [official source](https://pypi.python.org/pypi/pyrouge)).

Please write to us in case of 

### Evaluate as per ROUGE/BLEU mentioned in the paper
Since installation of ROUGE via the perl package is difficult, we use the other version of ROUGE (Details in main Repo). 

After the testing is complete, there will be a decode_test_* folder where * denotes the configuration details. Within this folder, there will be two folders *decoded* and *reference* respectively. Along with it a *results.txt* will be generated.

If this doesn't work, fetch the metrics folder from [main](https://github.com/nikitacs16/Holl-E) repository. Use the three files from that folder and run the following command

```
python evauluate.py path_to_decoded path_to_reference
```
This will output the BLEU, ROUGE-1, ROUGE-2, ROUGE-L on the screen.


### Help, I've got NaNs! 
Retaining this section from the previous README
For reasons that are [difficult to diagnose](https://github.com/abisee/pointer-generator/issues/4), NaNs sometimes occur during training, making the loss=NaN and sometimes also corrupting the model checkpoint with NaN values, making it unusable. Here are some suggestions:

* If training stopped with the `Loss is not finite. Stopping.` exception, you can just try restarting. It may be that the checkpoint is not corrupted.
* You can check if your checkpoint is corrupted by using the `inspect_checkpoint.py` script. If it says that all values are finite, then your checkpoint is OK and you can try resuming training with it.
* The training job is set to keep 3 checkpoints at any one time (see the `max_to_keep` variable in `run_summarization.py`). If your newer checkpoint is corrupted, it may be that one of the older ones is not. You can switch to that checkpoint by editing the `checkpoint` file inside the `train` directory.
* Alternatively, you can restore a "best model" from the `eval` directory. See the note **Restoring snapshots** above.
* If you want to try to diagnose the cause of the NaNs, you can run with the `--debug=1` flag turned on. This will run [Tensorflow Debugger](https://www.tensorflow.org/versions/master/programmers_guide/debugger), which checks for NaNs and diagnoses their causes during training.
