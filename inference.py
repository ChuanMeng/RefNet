from __future__ import print_function
import os
import tensorflow as tf
import greed_search
import data
import util
import evaluate
import json
import glob
import shutil

FLAGS = tf.app.flags.FLAGS


class Inference:
    """greed search decoder."""
    def __init__(self, model, batcher, vocab, ckpt_path):
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self.ckpt_path = ckpt_path
        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=util.get_config())

        self._saver.restore(self._sess, self.ckpt_path)
        print("load mode from %s" % self.ckpt_path)

        self.model_num = self.ckpt_path.split('-')[-1]
        ckpt_name = "ckpt-" + self.model_num  # this is something of the form "ckpt-123456"

        self._decode_dir = os.path.join(FLAGS.log_root, get_infer_dir(ckpt_name))

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir):
            os.mkdir(self._decode_dir)
        else:
            raise Exception("infer directory %s should not already exist")

    def infer(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return"""
        counter = 0
        output = {}
        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                print("Decoder has finished reading dataset for single_pass.")
                # log original information
                with open(os.path.join(self._decode_dir, "output.json"), 'w', encoding='utf-8') as w:
                    json.dump(output, w)
                print("Output has been saved in %s." % self._decode_dir)

                #start evaluation
                evaluate.main(self.ckpt_path, FLAGS.log_root, self._decode_dir, FLAGS.mode, FLAGS.multi_label_eval)
                return

            background_span = data.show_background_span(batch.original_backgrounds_token[0], batch.original_b_starts[0], batch.original_b_ends[0])
            response_span = data.show_background_span(batch.original_responses_token[0], batch.original_r_starts[0], batch.original_r_ends[0])
            # Run greed search to get best Hypothesis
            best_hyp = greed_search.run_greed_search(self._sess, self._model, self._vocab, batch)
            best_hyp.tokens = [token for token in best_hyp.tokens if token not in [None]]
            # Extract the output ids from the hypothesis and convert back to words
            output_ids = best_hyp.tokens[1:]
            decoded_token, highlights_decoded_token, spans = data.outputids2words(output_ids, self._vocab, batch.bac_oovs[0], batch.original_backgrounds_token[0])

            if output_ids[-1] == 3:
                output_ids_semantic = output_ids[:(len(output_ids)-1)]
            else:
                output_ids_semantic = output_ids

            ids_for_print = [str(i)for i in output_ids_semantic]
            ids_for_print = ' '.join(ids_for_print)

            switch_ref_probs = best_hyp.switch_ref_probs
            switch_ref_probs = [str(i) for i in switch_ref_probs]
            switch_ref_probs = ' '.join(switch_ref_probs)

            switch_gen_probs = best_hyp.switch_gen_probs
            switch_gen_probs = [str(i) for i in switch_gen_probs]
            switch_gen_probs = ' '.join(switch_gen_probs)

            switch_gen_pred_probs = best_hyp.switch_gen_pred_probs
            switch_gen_pred_probs = [str(i) for i in switch_gen_pred_probs]
            switch_gen_pred_probs = ' '.join(switch_gen_pred_probs)

            switch_gen_copy_probs = best_hyp.switch_gen_copy_probs
            switch_gen_copy_probs = [str(i) for i in switch_gen_copy_probs]
            switch_gen_copy_probs = ' '.join(switch_gen_copy_probs)

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_token.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                fst_stop_idx1 = highlights_decoded_token.index(data.STOP_DECODING)
                decoded_token = decoded_token[:fst_stop_idx]
                highlights_decoded_token = highlights_decoded_token[:fst_stop_idx1]

                if len(decoded_token) == 0:
                    decoded_token.append(".")

            except ValueError:
                decoded_token = decoded_token
                highlights_decoded_token = highlights_decoded_token

            spans_output = ' '.join(spans)
            decoded_output = ' '.join(decoded_token)
            highlights_decoded_output = ' '.join(highlights_decoded_token)

            output[batch.original_example_ids[0]] = {"background": background_span, "context": batch.original_contexts[0], "highlights_ref_response": response_span,
                                   "highlights_inferred_response": highlights_decoded_output, "ref_response": batch.original_responses[0],
                                   "inferred_response": decoded_output, "ref_span": batch.original_spans[0],"inferred_spans": spans_output, "output_index": output_ids_semantic,
                                   "switch_ref_probs": switch_ref_probs, "switch_gen_probs": switch_gen_probs,
                                   "switch_gen_pred_probs": switch_gen_pred_probs,"switch_gen_copy_probs": switch_gen_copy_probs}

            self.write_for_observation(batch.original_example_ids[0], background_span, batch.original_contexts[0], response_span, highlights_decoded_output, ids_for_print, switch_ref_probs, switch_gen_probs, switch_gen_pred_probs, switch_gen_copy_probs, counter)
            counter += 1  # this is how many examples we've decoded

    def write_for_observation(self, example_ids, background, contexts, ref_response, decoded_output, ids_for_print, switch_ref_probs, switch_gen_probs, switch_gen_pred_probs, switch_gen_copy_probs, ex_index):
        ref_file = os.path.join(self._decode_dir, "%s_%s_Inferred_Examples.txt" % (self.model_num, FLAGS.mode))
        with open(ref_file, "a", encoding="utf-8") as f:
            f.write("Example_ids:\n" + example_ids + "\n\n")
            f.write("Background:\n"+ background+"\n\n")
            f.write("Context:\n"+contexts + "\n\n")
            f.write("Reference_response:\n"+ ref_response + "\n\n")
            f.write("Inferenced_response:\n" + decoded_output+"\n\n")
            f.write("Ids_for_print:\n" + ids_for_print + "\n\n")
            f.write("Switch_Ref_Probs:\n" + switch_ref_probs + "\n\n")
            f.write("Switch_Gen_Probs:\n" + switch_gen_probs + "\n\n")
            f.write("Switch_Gen_Pred_Probs:\n" + switch_gen_pred_probs + "\n\n")
            f.write("Switch_Gen_Copy_Probs:\n" + switch_gen_copy_probs+ "\n\n\n\n")

        print("Wrote %s example %i to file" % (self.ckpt_path, ex_index))


def get_infer_dir(ckpt_name):
    if "val" in FLAGS.mode:
        dataset = "Validation"
    elif "test" in FLAGS.mode:
        dataset = "Test"

    dirname = "%s_Infer" % dataset
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
