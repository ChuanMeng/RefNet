"""This file contains code to process data into batches"""
import queue
from random import shuffle
from threading import Thread
import time
import numpy as np
from collections import namedtuple
import tensorflow as tf
import data


class Example:
    """Class representing a train/val/test example for response generation.
	"""
    def __init__(self, background_text, context_text, response_text, span_text, b_start, b_end, r_start, r_end,
                 example_id, vocab, hps):
        self.hps = hps
        self.b_start = int(b_start)
        self.b_end = int(b_end)
        self.r_start = int(r_start)
        self.r_end = int(r_end)
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the background
        background_token = background_text.split()

        if len(background_token) > hps.max_bac_enc_steps:
            background_token = background_token[:hps.max_bac_enc_steps]
            background_text = " ".join(b for b in background_token)
        self.background_len = len(background_token)
        self.background_input = [vocab.word2id(w) for w in background_token]  # list of word ids; OOVs are represented by the id for UNK token
        self.example_id = example_id

        # Process the context
        context_token = context_text.split()

        if len(context_token) > hps.max_con_enc_steps:
            context_token = context_token[len(context_token) - hps.max_con_enc_steps:]
            context_text = " ".join(c for c in context_token)
        self.context_len = len(context_token)  # store the length after truncation but before padding
        self.context_input = [vocab.word2id(w) for w in context_token]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the response
        response_token = response_text.split()
        response_ids = [vocab.word2id(w) for w in response_token]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(response_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
        # Store a version of the background_input where in-article OOVs are represented by their temporary OOV id;
        # also store the in-article OOVs words themselves
        self.background_input_extend_vocab, self.background_oovs = data.background2ids(background_token, vocab)
        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        response_ids_extend_vocab = data.response2ids(response_token, vocab, self.background_oovs)
        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(response_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

        # Store the original strings
        self.original_background_token = background_token
        self.original_background = background_text
        self.original_context = context_text
        self.original_response = response_text
        self.original_response_token = response_token
        self.original_span = span_text
        self.original_b_start = self.b_start
        self.original_b_end = self.b_end
        self.original_r_start = self.r_start
        self.original_r_end = self.r_end
        self.original_example_id = example_id


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_bac_encoder_input(self, max_len, pad_id):
        """Pad the background encoder input sequence with pad_id up to max_len."""
        while len(self.background_input) < max_len:
            self.background_input.append(pad_id)
        while len(self.background_input_extend_vocab) < max_len:
            self.background_input_extend_vocab.append(pad_id)

    def pad_con_encoder_input(self, max_len, pad_id):
        """Pad the context input sequence with pad_id up to max_len."""
        while len(self.context_input) < max_len:
            self.context_input.append(pad_id)


class Batch:
    """Class representing a minibatch of train/val/test examples for text summarization.
	"""
    def __init__(self, example_list, hps, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_bac_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_con_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)
        self.init_switch_label(example_list, hps)
        self.init_start_end_label(example_list, hps)
        self.store_orig_strings(example_list)  # store the original strings

    def init_bac_encoder_seq(self, example_list, hps):
        # Determine the maximum length of the encoder input sequence in this batch
        max_bac_encoder_seq_len = max([ex.background_len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_bac_encoder_input(max_bac_encoder_seq_len, self.pad_id)
        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.bac_enc_batch = np.zeros((hps.batch_size, max_bac_encoder_seq_len),dtype=np.int32)
        self.background_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.bac_enc_padding_mask = np.zeros((hps.batch_size, max_bac_encoder_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.bac_enc_batch[i, :] = ex.background_input[:]
            self.background_lens[i] = ex.background_len
            for j in range(ex.background_len):
                self.bac_enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        # Determine the max number of in-article OOVs in this batch
        self.max_bac_oovs = max([len(ex.background_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.bac_oovs = [ex.background_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.bac_enc_batch_extend_vocab = np.zeros((hps.batch_size, max_bac_encoder_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.bac_enc_batch_extend_vocab[i, :] = ex.background_input_extend_vocab[:]

    def init_con_encoder_seq(self, example_list, hps):
        # Determine the maximum length of the encoder input sequence in this batch
        max_con_encoder_seq_len = max([ex.context_len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_con_encoder_input(max_con_encoder_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.con_enc_batch = np.zeros((hps.batch_size, max_con_encoder_seq_len), dtype=np.int32)
        self.context_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.con_enc_padding_mask = np.zeros((hps.batch_size, max_con_encoder_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.con_enc_batch[i, :] = ex.context_input[:]
            self.context_lens[i] = ex.context_len
            for j in range(ex.context_len):
                self.con_enc_padding_mask[i][j] = 1

    def init_decoder_seq(self, example_list, hps):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        self.dec_switch_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                if (j > ex.r_start) and (j <= ex.r_end):
                    self.dec_switch_mask[i][j] = 0
                    self.dec_padding_mask[i][j] = 1
                else:
                    self.dec_switch_mask[i][j] = 1
                    self.dec_padding_mask[i][j] = 1

    def init_switch_label(self, example_list, hps):
        self.switch_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        for i, ex in enumerate(example_list):
            if ex.r_start < hps.max_dec_steps:
                self.switch_batch[i][ex.r_start] = 1
            else:
                continue

    def init_start_end_label(self, example_list, hps):
        max_bac_encoder_seq_len = max([ex.background_len for ex in example_list])
        self.bac_start_batch = np.zeros((hps.batch_size, max_bac_encoder_seq_len), dtype=np.float32)
        self.bac_end_batch = np.zeros((hps.batch_size, max_bac_encoder_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            if ex.b_start >= max_bac_encoder_seq_len:
                continue
            else:
                self.bac_start_batch[i][ex.b_start] = 1.0
                if ex.b_end >= max_bac_encoder_seq_len:
                    modified_b_end = max_bac_encoder_seq_len - 1
                    self.bac_end_batch[i][modified_b_end] = 1.0
                else:
                    self.bac_end_batch[i][ex.b_end] = 1.0


    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object """
        self.original_backgrounds_token = [ex.original_background_token for ex in example_list]
        self.original_backgrounds = [ex.original_background for ex in example_list]  # list of lists
        self.original_contexts = [ex.original_context for ex in example_list]  # list of lists
        self.original_responses = [ex.original_response for ex in example_list]
        self.original_responses_token = [ex.original_response_token for ex in example_list]
        self.original_spans = [ex.original_span for ex in example_list]
        self.original_b_starts = [ex.original_b_start for ex in example_list]
        self.original_b_ends = [ex.original_b_end for ex in example_list]
        self.original_r_starts = [ex.original_r_start for ex in example_list]
        self.original_r_ends = [ex.original_r_end for ex in example_list]
        self.original_example_ids = [ex.original_example_id for ex in example_list]


class Batcher:
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(
                Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()  # 启动线程活动。

    def next_batch(self):
        """Return a Batch from the batch queue.
		"""
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""

        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (background_text, context_text, response_text, span_text, b_start, b_end, r_start, r_end, example_id) = next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            example = Example(background_text, context_text, response_text, span_text, b_start, b_end, r_start, r_end, example_id, self._vocab, self._hps)

            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.
		"""
        while True:
            if self._hps.mode == 'train':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(
                        self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.background_len)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # greed search inference mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            # 一个
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        """Generates article and abstract text from tf.Example.
		Args:
			example_generator: a generator of tf.Examples from file. See data.example_generator"""
        while True:
            e = next(example_generator)
            try:
                background_text = e.features.feature['background'].bytes_list.value[0].decode()
                context_text = e.features.feature['context'].bytes_list.value[0].decode()
                response_text = e.features.feature['response'].bytes_list.value[0].decode()
                span_text = e.features.feature['span'].bytes_list.value[0].decode()
                b_start = e.features.feature['b_start'].bytes_list.value[0].decode()
                b_end = e.features.feature['b_end'].bytes_list.value[0].decode()
                r_start = e.features.feature['r_start'].bytes_list.value[0].decode()
                r_end = e.features.feature['r_end'].bytes_list.value[0].decode()
                example_id = e.features.feature['example_id'].bytes_list.value[0].decode()

            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(background_text) == 0:
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield (background_text, context_text, response_text, span_text, b_start, b_end, r_start, r_end, example_id)


if __name__ == '__main__':

    hps_dict = {'mode':'train', 'batch_size': 16, 'max_bac_enc_steps': 300,'max_con_enc_steps': 65, 'max_dec_steps': 95}
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    vocab = data.Vocab('data/mixed_context/finished_files/vocab', 25000)
    batcher = Batcher('data/mixed_context/finished_files/chunked/train_*', vocab, hps, single_pass=False)
    batch = batcher.next_batch()

    # print("batch.target_batch: ",batch.target_batch)
    i = 0
    print()
    print("backgrounds: ", batch.original_backgrounds[i], "\n")
    print("contexts: ", batch.original_contexts[i], "\n")
    print("responses: ", batch.original_responses[i], "\n")
    print("spans: ", batch.original_spans[i], "\n")
    print("b_starts: ", batch.original_b_starts[i], "\n")
    print("b_ends: ", batch.original_b_ends[i], "\n")
    print("r_starts: ", batch.original_r_starts[i], "\n")
    print("r_ends: ", batch.original_r_ends[i], "\n")
    print("example_ids: ", batch.original_example_ids[i], "\n")

    print("batch.dec_padding_mask: ", batch.dec_padding_mask[i], "\n")
    print("batch.switch_mask: ", batch.dec_switch_mask[i], "\n")
    print("batch.switch_batch: ", batch.switch_batch[i], "\n")
    print("batch.bac_start_batch: ", batch.bac_start_batch[i], "\n")
    print("batch.bac_end_batch: ", batch.bac_end_batch[i], "\n")

    batch = batcher.next_batch()
    print("======================================================----------")
    i = 0
    print()
    print("backgrounds: ", batch.original_backgrounds[i], "\n")
    print("contexts: ", batch.original_contexts[i], "\n")
    print("responses: ", batch.original_responses[i], "\n")
    print("spans: ", batch.original_spans[i], "\n")
    print("b_starts: ", batch.original_b_starts[i], "\n")
    print("b_ends: ", batch.original_b_ends[i], "\n")
    print("r_starts: ", batch.original_r_starts[i], "\n")
    print("r_ends: ", batch.original_r_ends[i], "\n")
    print("example_ids: ", batch.original_example_ids[i], "\n")

    print("batch.dec_padding_mask: ", batch.dec_padding_mask[i], "\n")
    print("batch.switch_mask: ", batch.dec_switch_mask[i], "\n")
    print("batch.switch_batch: ", batch.switch_batch[i], "\n")
    print("batch.bac_start_batch: ", batch.bac_start_batch[i], "\n")
    print("batch.bac_end_batch: ", batch.bac_end_batch[i], "\n")
