import tensorflow as tf
import data

FLAGS = tf.app.flags.FLAGS


class Hypothesis:
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, probs, state, attn_dists, switch_ref_probs, switch_gen_probs, switch_gen_pred_probs, switch_gen_copy_probs):

        self.tokens = tokens
        self.probs = probs
        self.state = state
        self.attn_dists = attn_dists
        self.switch_ref_probs = switch_ref_probs
        self.switch_gen_probs = switch_gen_probs
        self.switch_gen_pred_probs = switch_gen_pred_probs
        self.switch_gen_copy_probs = switch_gen_copy_probs

    def extend(self, token, prob, state, attn_dist, switch_ref_prob, switch_gen_prob, switch_gen_pred_prob, switch_gen_copy_prob):

        return Hypothesis(tokens=self.tokens + [token],
                          probs=self.probs + [prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          switch_ref_probs=self.switch_ref_probs + [switch_ref_prob],
                          switch_gen_probs=self.switch_gen_probs + [switch_gen_prob],
                          switch_gen_pred_probs=self.switch_gen_pred_probs + [switch_gen_pred_prob],
                          switch_gen_copy_probs=self.switch_gen_copy_probs + [switch_gen_copy_prob])


    @property
    def latest_token(self):
        return self.tokens[-1]


def run_greed_search(sess, model, vocab, batch):

    enc_batch, enc_states, que_states, dec_in_state = model.run_encoder(sess, batch)
    hyp = Hypothesis(tokens=[vocab.word2id(data.START_DECODING)], probs=[], state=dec_in_state, attn_dists=[], switch_ref_probs=[], switch_gen_probs=[], switch_gen_pred_probs=[], switch_gen_copy_probs=[])

    steps = 0
    while True:
        latest_token = hyp.latest_token
        if isinstance(latest_token, list):
            span_length = latest_token[1]-latest_token[0]+1
            mask_lenth = span_length - 1
            for i in range(mask_lenth):
                mask_one_token = [[enc_batch[0][latest_token[0]+i]]]
                state = hyp.state
                (_, _, _, _, _, _, _,_,_,new_state) = model.inference_step(sess=sess, batch=batch, latest_tokens=mask_one_token, bac_states=enc_states, que_states=que_states, dec_init_states=state)
                hyp = hyp.extend(token=None, prob=None, state=new_state, attn_dist="<mask>", switch_ref_prob="<mask>", switch_gen_prob="<mask>", switch_gen_pred_prob="<mask>", switch_gen_copy_prob="<mask>")
            latest_token = [[enc_batch[0][latest_token[1]]]]
        else:
            latest_token = [[latest_token if latest_token in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN)]]

        state = hyp.state
        # Run one step of the decoder to get the new info
        (word_ids, word_probs, span_ids, span_probs, switch_ref_prob, switch_gen_prob, switch_gen_pred_prob, switch_gen_copy_prob, attn_dist, new_state) = model.inference_step(sess=sess, batch=batch, latest_tokens=latest_token, bac_states=enc_states, que_states=que_states, dec_init_states=state)

        # span level
        if switch_ref_prob >= switch_gen_prob:
            token = span_ids
            prob = span_probs
            step = span_ids[1]-span_ids[0] + 1
        # word level
        else:
            token = word_ids  # int
            prob = word_probs  # float
            step = 1

        # Extend each hypothesis and collect them all in all_hyps

        hyp = hyp.extend(token=token, prob=prob, state=new_state, attn_dist=attn_dist, switch_ref_prob=switch_ref_prob, switch_gen_prob=switch_gen_prob, switch_gen_pred_prob=switch_gen_pred_prob, switch_gen_copy_prob=switch_gen_copy_prob)

        steps += step
        # Filter and collect any hypotheses that have produced the end token.
        if hyp.latest_token == vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
            break
        if steps >= FLAGS.max_dec_steps:
            break

    # Return the hypothesis with highest average log prob
    return hyp

