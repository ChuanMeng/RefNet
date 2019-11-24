import time
import numpy as np
import tensorflow as tf
from hybrid_decoder import hybrid_decoder
import util

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # background encoder part
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='background_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='background_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='background_padding_mask')

        self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='background_batch_extend_vocab')
        self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # context encoder part
        self._que_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='context_batch')
        self._que_lens = tf.placeholder(tf.int32, [hps.batch_size], name='context_lens')
        self._que_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='context_padding_mask')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
        self._dec_switch_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_switch_mask')

        # train label part
        self._bac_start_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='bac_start_batch')
        self._bac_end_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='bac_end_batch')
        self._switch_batch = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='switch_batch')

    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict ={}
        feed_dict[self._enc_batch] = batch.bac_enc_batch
        feed_dict[self._enc_lens] = batch.background_lens
        feed_dict[self._enc_padding_mask] = batch.bac_enc_padding_mask

        feed_dict[self._que_batch] = batch.con_enc_batch
        feed_dict[self._que_lens] = batch.context_lens
        feed_dict[self._que_padding_mask] = batch.con_enc_padding_mask

        feed_dict[self._enc_batch_extend_vocab] = batch.bac_enc_batch_extend_vocab
        feed_dict[self._max_art_oovs] = batch.max_bac_oovs

        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch  # batch_size*decoder_max_time_step
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
            feed_dict[self._dec_switch_mask] = batch.dec_switch_mask
            feed_dict[self._switch_batch] = batch.switch_batch
            feed_dict[self._bac_start_batch] = batch.bac_start_batch
            feed_dict[self._bac_end_batch] = batch.bac_end_batch

        return feed_dict

    def _add_backgroud_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('background_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(encoder_outputs, 2)
        return encoder_outputs, fw_st, bw_st

    def _add_context_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('context_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(encoder_outputs, 2)  # bz*timestep*2h
        return encoder_outputs, fw_st, bw_st

    def _add_matching_layer(self, bac_encoder_inputs, con_encoder_inputs, bac_seq_len, gate=None):
        with tf.variable_scope('matching_layer'):
            background_max_len = tf.shape(bac_encoder_inputs)[1]
            context_max_len = tf.shape(con_encoder_inputs)[1]

            expanded_context = tf.tile(tf.expand_dims(con_encoder_inputs, -3), (1, background_max_len, 1, 1)) # (batch_size, max_nodes, query_len, node_feature_dim)
            expanded_background = tf.tile(tf.expand_dims(bac_encoder_inputs, -2), (1, 1, context_max_len, 1)) #  (batch_size, max_nodes, query_len, node_feature_dim)
            dot_product_matrix = expanded_background * expanded_context
            concat_similarity_matrix = tf.concat((expanded_background, expanded_context, dot_product_matrix), -1)
            similarity_matrix = tf.reduce_mean(util.dense(concat_similarity_matrix, 1, use_bias=False, scope="similarity_matrix"), -1)  # (batch_size, max_nodes, max_query)

            # mask similarity_matrix
            context_mask = tf.tile(tf.expand_dims(self._que_padding_mask, axis=1), [1, background_max_len, 1])  # Tensor shape(batch * bac_len * con_len )
            context_masked_similarity_matrix = util.mask_softmax(context_mask, similarity_matrix)   # Tensor shape(batch * bac_len * con_len )

            # background2context
            similarity_matrix_softmax = tf.nn.softmax(context_masked_similarity_matrix, -1)  # Tensor shape(batch, bac_len, con_len)
            background2context = tf.matmul(similarity_matrix_softmax, con_encoder_inputs)  # Tensor shape(batch, bac_len, 2hz)

            # context2background
            background_mask = self._enc_padding_mask  # Tensor shape(batch * bac_len)
            squeezed_context_masked_similarity_matrix = tf.reduce_max(context_masked_similarity_matrix, -1)  # Tensor shape(batch * bac_len)
            background_masked_similarity_matrix = util.mask_softmax(background_mask, squeezed_context_masked_similarity_matrix)  # Tensor shape(batch * bac_len)
            b = tf.nn.softmax(background_masked_similarity_matrix, -1)  # Tensor shape(batch * bac_len)
            context2background = tf.matmul(tf.expand_dims(b, 1), bac_encoder_inputs)  # (batch_size,1,bac_len) (batch_size, bac_len, feature_dim) = (batch_size,1,2hz)
            context2background = tf.tile(context2background, (1, background_max_len, 1))  #  (batch_size,background_max_len, 2hz)
            G = tf.concat((bac_encoder_inputs, background2context, bac_encoder_inputs * background2context, bac_encoder_inputs * context2background), -1)

        with tf.variable_scope('modeling_layer1'):
            cell_fw_1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw_1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs1, (fw_st1, bw_st1)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_1, cell_bw_1, G, dtype=tf.float32, sequence_length= bac_seq_len, swap_memory=True)
            matching_output1 = tf.concat(encoder_outputs1, 2)  # Tensor shape(batch, bac_len, 2*hz)

        with tf.variable_scope('modeling_layer2'):
            cell_fw_2 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw_2 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs2, (fw_st2, bw_st2)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_2, cell_bw_2, matching_output1, dtype=tf.float32, sequence_length= bac_seq_len, swap_memory=True)
            matching_output2 = tf.concat(encoder_outputs2, 2)  # Tensor shape(batch, bac_len, 2*hz)
        return matching_output2, fw_st2, bw_st2

    def _reduce_states(self, fw_st, bw_st, fw_st_q, bw_st_q):
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 4, hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 4, hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat([fw_st.c, bw_st.c, fw_st_q.c, bw_st_q.c], 1)  # Concatenation of fw and bw cell
            old_h = tf.concat([fw_st.h, bw_st.h, fw_st_q.h, bw_st_q.h], 1)  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, inputs):
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        outputs, out_state, attn_dists, switch_ref_time_step, switch_gen_time_step, switch_gen_pred_time_step, switch_gen_copy_time_step = hybrid_decoder(inputs,
                                                                                                               self._dec_in_state,
                                                                                                               self._background_final_state,
                                                                                                               self._enc_padding_mask,
                                                                                                               self._que_states,
                                                                                                               self._que_padding_mask,
                                                                                                               cell,
                                                                                                               initial_state_attention=(hps.mode in ["test","val"]))

        return outputs, out_state, attn_dists, switch_ref_time_step, switch_gen_time_step, switch_gen_pred_time_step, switch_gen_copy_time_step

    def _calc_word_level_dist(self, vocab_dists, attn_dists):
        with tf.variable_scope('calc_word_level_dist'):
            vocab_dists = [switch_gen_pred_one_step * dist for (switch_gen_pred_one_step, dist) in zip(self.switch_gen_pred_time_step, vocab_dists)]
            attn_dists = [switch_gen_copy_one_step * dist for (switch_gen_copy_one_step, dist) in zip(self.switch_gen_copy_time_step, attn_dists)]

            extended_vsize = self._vocab.size() + self._max_art_oovs
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))  # batch_size*max_art_oovs
            vocab_dists_extended = [tf.concat([dist, extra_zeros], 1) for dist in
                                    vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=self._hps.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over encode
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [self._hps.batch_size, extended_vsize]  # 画布
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            word_level_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return word_level_dists  # [(batch_size, extended_vsize) ,(batch_size, extended_vsize) ...]

    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=self.trunc_norm_init)

                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_que_inputs = tf.nn.embedding_lookup(embedding, self._que_batch)  # tensor with shape (batch_size, max_que_steps, emb_size)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

            # Add the backgrpind encoder.
            enc_outputs, fw_st_b, bw_st_b = self._add_backgroud_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            # Add the context encoder.
            que_outputs, fw_st_q, bw_st_q = self._add_context_encoder(emb_que_inputs, self._que_lens)
            self._que_states = que_outputs

            # Add matching layer
            if self._hps.matching_layer is True:
                matching_outputs, fw_st_m, bw_st_m = self._add_matching_layer(self._enc_states, self._que_states, self._enc_lens, gate=True)
                self._matching_states = matching_outputs  # Tensor shape(batch*bac_len*2hz)
                self._background_final_state = self._matching_states
                fw_st = fw_st_m
                bw_st = bw_st_m

            else:
                self._background_final_state = self._enc_states
                fw_st = fw_st_b
                bw_st = bw_st_b

            # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
            self._dec_in_state = self._reduce_states(fw_st, bw_st, fw_st_q, bw_st_q)

            # Add the decoder.
            with tf.variable_scope('hybrid_decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.switch_ref_time_step, self.switch_gen_time_step, self.switch_gen_pred_time_step, self.switch_gen_copy_time_step = self._add_decoder(emb_dec_inputs)

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('generation_decoding'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))

                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
                # calc final distribution from copy distribution and vocabulary distribution
                self.word_level_dists = self._calc_word_level_dist(vocab_dists, self.attn_dists)

            with tf.variable_scope('reference_decoding'):
                # v^T tanh(W_b b_i + W_o output_t + b_attn)
                self.start_dist = []
                self.end_dist = []

                background_states = self._background_final_state  # [batch_size x max_encode_length x 2hidden_size]
                encode_state_length = background_states.get_shape()[2].value  # 2hidden_size
                attention_vec_size = encode_state_length  # 2hidden_size

                w_b = tf.get_variable("W_b", [encode_state_length, attention_vec_size])

                w_step = tf.get_variable('W_step', [hps.hidden_dim, attention_vec_size], dtype=tf.float32,initializer=self.trunc_norm_init)
                bias_step = tf.get_variable('bias_step', [attention_vec_size], dtype=tf.float32, initializer=self.trunc_norm_init)

                v = tf.get_variable("v", [attention_vec_size])
                background_features = tf.einsum("ijk,kl->ijl", background_states, w_b)  # shape (batch_size,max_encode_length,attention_vec_size)

                if hps.multi_hop_span_pre_mode == 'mlp':
                    w_mlp = tf.get_variable('W_mlp', [3 * hps.hidden_dim, hps.hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
                    bias_mlp = tf.get_variable('bias_mlp', [hps.hidden_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
                    for i, hop_1 in enumerate(decoder_outputs):
                        #start step
                        hop_1_features = tf.nn.xw_plus_b(hop_1, w_step, bias_step)  # shape (batch_size,attention_vec_size)
                        hop_1_features = tf.expand_dims(hop_1_features, 1)  # shape (batch_size,1, attention_vec_size)
                        start_dist = tf.reduce_sum(v * tf.tanh(background_features + hop_1_features), 2)  # (batch_size,max_encode_length)
                        start_dist = tf.nn.softmax(util.mask_softmax(self._enc_padding_mask, start_dist))  # take softmax. shape (batch_size, max_encode_length)
                        self.start_dist.append(start_dist)

                        start_dist_ex_dim = tf.expand_dims(start_dist, 2) # shape (batch_size, max_encode_length, 1)
                        start_vector = tf.reduce_sum(start_dist_ex_dim * background_states, 1)  # shape (batch_size, * 2hidden_size).
                        start_vector = tf.reshape(start_vector, [-1, encode_state_length]) # shape (batch_size, * 2hidden_size).

                        #end_step
                        concat_vector = tf.concat([hop_1, start_vector], 1) #batch_size*3hidden_size
                        hop_2 = tf.nn.xw_plus_b(concat_vector, w_mlp, bias_mlp)  #batch_size*hidden_size

                        hop_2_features = tf.nn.xw_plus_b(hop_2, w_step, bias_step) # shape (batch_size,attention_vec_size )
                        hop_2_features = tf.expand_dims(hop_2_features, 1) #shape(batch_size,1, attention_vec_size)
                        end_dist = tf.reduce_sum(v * tf.tanh(background_features + hop_2_features), 2)
                        end_dist = tf.nn.softmax(util.mask_softmax(self._enc_padding_mask, end_dist))
                        self.end_dist.append(end_dist)

                elif hps.multi_hop_span_pre_mode == 'rnn':
                    cell_pre_span = tf.nn.rnn_cell.GRUCell(hps.hidden_dim, kernel_initializer=self.rand_unif_init)
                    for i, hop_1 in enumerate(decoder_outputs):
                        initial_state = hop_1
                        hop_1_features = tf.nn.xw_plus_b(initial_state, w_step, bias_step)  # shape (batch_size,attention_vec_size)
                        hop_1_features = tf.expand_dims(hop_1_features, 1)  # shape (batch_size,1, attention_vec_size)
                        start_dist = tf.reduce_sum(v * tf.tanh(background_features + hop_1_features),2)  # (batch_size,max_encode_length)
                        start_dist = tf.nn.softmax(util.mask_softmax(self._enc_padding_mask,start_dist))  # take softmax. shape (batch_size, max_encode_length)
                        self.start_dist.append(start_dist)

                        start_dist_ex_dim = tf.expand_dims(start_dist, 2)  # shape (batch_size, max_encode_length, 1)
                        start_vector = tf.reduce_sum(start_dist_ex_dim * background_states, 1)  # shape (batch_size, * 2hidden_size).
                        start_vector = tf.reshape(start_vector, [-1, encode_state_length])  # shape (batch_size, * 2hidden_size).

                        output, state = cell_pre_span(start_vector, initial_state)

                        hop_2_features = tf.nn.xw_plus_b(state, w_step, bias_step)  # shape (batch_size,attention_vec_size)
                        hop_2_features = tf.expand_dims(hop_2_features, 1)  # shape(batch_size,1, attention_vec_size)
                        end_dist = tf.reduce_sum(v * tf.tanh(background_features + hop_2_features), 2)
                        end_dist = tf.nn.softmax(util.mask_softmax(self._enc_padding_mask, end_dist))
                        self.end_dist.append(end_dist)

            with tf.variable_scope('train_loss'):
                if hps.mode == 'train':
                    # Calculate the loss
                    self.gen_mode_work_num = tf.cast(tf.count_nonzero(self._dec_padding_mask), tf.float32)
                    self.switch_work_num = tf.cast(tf.count_nonzero(self._dec_switch_mask), tf.float32)
                    self.ref_mode_work_num = tf.cast(tf.count_nonzero(self._bac_start_batch), tf.float32) + tf.cast(tf.count_nonzero(self._bac_end_batch), tf.float32)

                    with tf.variable_scope('switch_loss'):
                        switch_gen_matrix = tf.reshape(tf.transpose(tf.convert_to_tensor(self.switch_gen_time_step), perm=[1, 0, 2]), [hps.batch_size, hps.max_dec_steps])
                        switch_ref_matrix = tf.reshape(tf.transpose(tf.convert_to_tensor(self.switch_ref_time_step), perm=[1, 0, 2]), [hps.batch_size, hps.max_dec_steps])
                        switch_ref_loss = - tf.reduce_sum(self._switch_batch * tf.log(switch_ref_matrix + 1e-10) * self._dec_switch_mask)
                        switch_gen_loss = - tf.reduce_sum((1 - self._switch_batch) * tf.log(switch_gen_matrix + 1e-10) * self._dec_switch_mask)
                        self.switch_loss = (switch_ref_loss + switch_gen_loss) / self.switch_work_num

                    with tf.variable_scope('generation_loss'):
                        word_level_dists = tf.convert_to_tensor(self.word_level_dists)
                        word_level_dists = tf.transpose(word_level_dists, perm=[1, 0, 2])  # batch * decoder_max_len * (vocab_size + OOV_size)
                        word_level_outputs_one_hot = tf.one_hot(self._target_batch, vsize + self._max_art_oovs)
                        word_level_crossent = - tf.reduce_sum(word_level_outputs_one_hot * tf.log(word_level_dists + 1e-10),-1)
                        self.generation_loss = tf.reduce_sum(word_level_crossent * self._dec_padding_mask) / self.gen_mode_work_num

                    with tf.variable_scope('reference_loss'):
                        start_dist_matrix = tf.transpose(tf.convert_to_tensor(self.start_dist), perm=[1, 0, 2])  # batch * max_dec_steps * max_encode_length
                        end_dist_matrix = tf.transpose(tf.convert_to_tensor(self.end_dist), perm=[1, 0, 2])  # batch * max_dec_steps * max_encode_length

                        start_label = tf.expand_dims(self._bac_start_batch, 1)  # batch * 1* max_encode_length
                        end_label = tf.expand_dims(self._bac_end_batch, 1)  # batch * 1* max_encode_length
                        start_loss_all_step = - tf.reduce_sum(start_label * tf.log(start_dist_matrix + 1e-10), -1)
                        end_losss_all_step = - tf.reduce_sum(end_label * tf.log(end_dist_matrix + 1e-10), -1) # batch * max_dec_steps
                        start_loss = tf.reduce_sum(start_loss_all_step * self._switch_batch)
                        end_loss = tf.reduce_sum(end_losss_all_step * self._switch_batch)
                        switch_adhere_loss = - tf.reduce_sum(self._switch_batch * tf.log(switch_ref_matrix + 1e-10) * self._dec_switch_mask)

                        self.reference_loss = (start_loss + end_loss + switch_adhere_loss) / self.ref_mode_work_num

                    with tf.variable_scope('total_loss'):
                        self.total_loss = self.switch_loss + self.generation_loss + self.reference_loss
                        tf.summary.scalar('total_loss', self.total_loss)

            with tf.variable_scope('inference'):
                if hps.mode in ['val','test']:
                    # We run inference greed search mode one decoder step or multi decoder steps at a time
                    # generation mode
                    infer_word_level_dists = self.word_level_dists[0]
                    self.word_probs, self.word_ids = tf.nn.top_k(infer_word_level_dists, 1)  # take the 1 #1*1

                    #reference mode
                    outer = tf.matmul(tf.expand_dims(self.start_dist[0], 2), tf.expand_dims(self.end_dist[0], axis=1)) # shape(batch * bac_len * bac_len)
                    outer = tf.matrix_band_part(outer, 0, hps.max_span_len)
                    self.start_prob, self.start_index = tf.nn.top_k(tf.reduce_max(outer, 2), 1) # shape(batch*l_start)=> batch*1
                    self.end_prob, self.end_index = tf.nn.top_k(tf.reduce_max(outer, 1), 1) # shape(batch*l_end)=>  batch*1

                    #switcher
                    self.infer_switch_ref = self.switch_ref_time_step[0]  # 1*1
                    self.infer_switch_gen = self.switch_gen_time_step[0]  # 1*1
                    self.infer_switch_gen_pred = self.switch_gen_pred_time_step[0]  # 1*1
                    self.infer_switch_gen_copy = self.switch_gen_copy_time_step[0]  # 1*1

                    self.infer_attn_dists = self.attn_dists[0]

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        hps = self._hps
        self._lr = tf.Variable(hps.lr, trainable=False, name='learning_rate')
        loss_to_minimize = self.total_loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step ."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'total_loss': self.total_loss,
            'switch_loss': self.switch_loss,
            'generation_loss': self.generation_loss,
            'reference_loss': self.reference_loss,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
        (enc_batch, bac_states, que_states, dec_in_state, global_step) = sess.run([self._enc_batch, self._background_final_state, self._que_states, self._dec_in_state, self.global_step], feed_dict)  # run the encoder
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c, dec_in_state.h)
        return enc_batch, bac_states, que_states, dec_in_state

    def inference_step(self, sess, batch, latest_tokens, bac_states, que_states, dec_init_states):
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_init_states.c, dec_init_states.h)

        feed = {
            self._background_final_state: bac_states,
            self._que_states: que_states,
            self._enc_padding_mask: batch.bac_enc_padding_mask,
            self._que_padding_mask: batch.con_enc_padding_mask,
            self._enc_batch_extend_vocab: batch.bac_enc_batch_extend_vocab,
            self._max_art_oovs: batch.max_bac_oovs,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.array(latest_tokens)
        }

        to_return = {
            "word_ids": self.word_ids,
            "word_probs": self.word_probs,

            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_prob": self.start_prob,
            "end_prob": self.end_prob,

            "switch_ref": self.infer_switch_ref,
            "switch_gen": self.infer_switch_gen,
            "switch_gen_pred": self.infer_switch_gen_pred,
            "switch_gen_copy": self.infer_switch_gen_copy,

            "attn_dists": self.infer_attn_dists,
            "states": self._dec_out_state
        }
        results = sess.run(to_return, feed_dict=feed)  # infer step
        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        word_ids = results['word_ids'][0].tolist()[0]
        word_probs = results['word_probs'][0].tolist()[0]
        span_ids = [results['start_index'].tolist()[0][0], results['end_index'].tolist()[0][0]]
        span_probs = [results['start_prob'].tolist()[0][0], results['end_prob'].tolist()[0][0]]

        switch_ref_prob = results['switch_ref'][0].tolist()[0]
        switch_gen_prob = results['switch_gen'][0].tolist()[0]
        switch_gen_pred_prob = results['switch_gen_pred'][0].tolist()[0]
        switch_gen_copy_prob = results['switch_gen_copy'][0].tolist()[0]

        attn_dists = results['attn_dists'][0].tolist()
        new_states = tf.contrib.rnn.LSTMStateTuple(results['states'].c, results['states'].h)

        return word_ids, word_probs, span_ids, span_probs, switch_ref_prob, switch_gen_prob, switch_gen_pred_prob, switch_gen_copy_prob, attn_dists, new_states



