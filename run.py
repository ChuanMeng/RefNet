import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import Model
from inference import Inference
import util
import yaml
import json
import time


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config_file', 'config.yaml', 'pass the config_file through command line if new expt')
config = yaml.load(open(FLAGS.config_file, 'r'))
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_device_id']

tf.app.flags.DEFINE_string('mode', config['mode'], 'must be one of train/test')
tf.app.flags.DEFINE_string('train_path', config['train_path'], 'Default path to the chunked files')
tf.app.flags.DEFINE_string('dev_path', config['dev_path'], 'Default path to the chunked files')
tf.app.flags.DEFINE_string('test_path', config['test_path'], 'Default path to the chunked files')
tf.app.flags.DEFINE_string('vocab_path', config['vocab_path'], 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('appoint_test', None, 'appoint a model to test')
# storage
tf.app.flags.DEFINE_string('log_root', config['log_root'], 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', config['exp_name'],'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('epoch_num', config['epoch_num'], 'the max num of train epoch num')
tf.app.flags.DEFINE_integer('hidden_dim', config['hidden_dim'], 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', config['emb_dim'], 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', config['batch_size'], 'minibatch size')
tf.app.flags.DEFINE_integer('max_bac_enc_steps', config['max_bac_enc_steps'], 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_con_enc_steps', config['max_con_enc_steps'],'max timesteps of query encoder (max source query tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', config['max_dec_steps'], 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('vocab_size', config['vocab_size'],'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', config['lr'], 'learning rate')
tf.app.flags.DEFINE_float('rand_unif_init_mag', config['rand_unif_init_mag'], 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', config['trunc_norm_init_std'], 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', config['max_grad_norm'], 'for gradient clipping')
tf.app.flags.DEFINE_integer('max_span_len', config['max_span_len'], 'the max length of predicted span')
tf.app.flags.DEFINE_string('multi_hop_span_pre_mode', config['multi_hop_span_pre_mode'], 'the mode of muilti_hop_span prediction.[rnn|mlp]')
tf.app.flags.DEFINE_bool('multi_label_eval', config['multi_label_eval'], 'do multi_label_evalation for testset, only for test')
tf.app.flags.DEFINE_bool('matching_layer', config['matching_layer'], 'whether use matching layer or not ')
tf.app.flags.DEFINE_bool('matching_gate', config['matching_gate'], 'whether use gate in matching layer')


def train(model, batcher):
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print("Preparing or waiting for session...")
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=FLAGS.epoch_num)
    print("Created session.")
    sess = tf.Session(config=util.get_config())
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    resume = True
    if resume:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            last_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, last_global_step is %s' % last_global_step)
        else:
            print('No checkpoint file found')

    print("starting run_training")

    train_set_num = 34486  # The number of examples in training set
    log_epoch = {}
    steps_for_one_epoch = int(train_set_num / FLAGS.batch_size)

    while True:
        batch = batcher.next_batch()
        print('Training: %s' % FLAGS.exp_name)
        t0 = time.time()
        results = model.run_train_step(sess, batch)
        t1 = time.time()

        total_loss = results['total_loss']
        switch_loss = results['switch_loss']
        generation_loss = results['generation_loss']
        reference_loss = results['reference_loss']
        global_step = results['global_step']

        epoch = int(global_step/steps_for_one_epoch)

        if global_step % steps_for_one_epoch == 0:
            log_epoch[epoch] = int(global_step)
            saver.save(sess, os.path.join(train_dir, "model.ckpt"), global_step=global_step)

        if epoch == FLAGS.epoch_num:
            print("Compele %d epochs, train is finished" % FLAGS.epoch_num)
            file = open(os.path.join(train_dir, "log_epoch.json"), 'w', encoding='utf-8')
            json.dump(log_epoch, file)
            file.close()
            break

        print('Epoch:%d, Step:%d, Train Loss: %f, Switch_Loss: %f, Generation_Loss: %f, Reference_Loss: %f, Seconds: %.3f' % (epoch, global_step, total_loss, switch_loss, generation_loss, reference_loss, t1 - t0))

        if not np.isfinite(total_loss):
            raise Exception("Loss is not finite. Stopping.")

        summaries = results['summaries']
        summary_writer.add_summary(summaries, global_step)  # write the summaries
        if global_step % 100 == 0:  # flush the summary writer every so often
            summary_writer.flush()

    sess.close()


def main(unused_argv):
    global config

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    print('Starting %s in %s mode...' % (FLAGS.exp_name, FLAGS.mode))

    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    hparam_list = ['mode', 'lr', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm','hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps','max_bac_enc_steps', 'max_con_enc_steps', 'max_span_len', 'multi_hop_span_pre_mode', 'matching_layer', 'matching_gate']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val

    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    tf.set_random_seed(111)

    if hps.mode == 'train':
        batcher = Batcher(FLAGS.train_path, vocab, hps, single_pass=False)
        print("creating model...")
        model = Model(hps, vocab)
        train(model, batcher)

    elif hps.mode == 'val':
        train_dir = os.path.join(FLAGS.log_root, "train")
        hps = hps._replace(batch_size=1)
        infer_model_hps = hps._replace(max_dec_steps=1)

        try:
            r = open(os.path.join(train_dir, "finished_val_models.json"), 'r', encoding='utf-8')
            finished_val_models = json.load(r)
            r.close()
        except FileNotFoundError:
            finished_val_models = {"finished_val_models": []}
            w = open(os.path.join(train_dir, "finished_val_models.json"), 'w', encoding='utf-8')
            json.dump(finished_val_models, w)
            w.close()

        while True:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                for ckpt_path in list(ckpt.all_model_checkpoint_paths):
                    if ckpt_path in finished_val_models["finished_val_models"]:
                        print("Val_mode: %s already has been evaluated, skip it" % ckpt_path)
                        pass
                    else:
                        print("Val_mode: start new eval %s" % ckpt_path)
                        batcher = Batcher(FLAGS.dev_path, vocab, hps, single_pass=True)
                        model = Model(infer_model_hps, vocab)
                        val_infer = Inference(model, batcher, vocab, ckpt_path)
                        val_infer.infer()
                        tf.reset_default_graph()
                        finished_val_models["finished_val_models"].append(ckpt_path)
                        w = open(os.path.join(train_dir, "finished_val_models.json"), 'w', encoding='utf-8')
                        json.dump(finished_val_models, w)
                        w.close()
                        print("Val_mode: finished one eval %s" % ckpt_path)
                print("Val_mode: current iterations of all_model_checkpoint_paths are completed...")
                print("Val_mode: finished %d modes" % len(finished_val_models["finished_val_models"]))
                if len(finished_val_models["finished_val_models"]) == FLAGS.epoch_num:
                    print("All val is ended")
                    break
            else:
                print('Val_mode: wait train finish the first epoch...')
            time.sleep(60)

    elif hps.mode == 'test':
        hps = hps._replace(batch_size=1)
        batcher = Batcher(FLAGS.test_path, vocab, hps, single_pass=True)
        infer_model_hps = hps._replace(max_dec_steps=1)
        model = Model(infer_model_hps, vocab)
        if FLAGS.test_model_dir is None:
            raise Exception("should appoint the test_model_dir")
        test_infer = Inference(model, batcher, vocab, FLAGS.test_model_dir)
        test_infer.infer()

    else:
        raise ValueError("The 'mode' flag must be one of train/val/test")


if __name__ == '__main__':
    tf.app.run()
