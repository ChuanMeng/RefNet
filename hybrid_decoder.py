import tensorflow as tf
import util


def hybrid_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, query_states, que_padding_mask, cell, initial_state_attention=False):
    with tf.variable_scope("attention_decoder"):
        batch_size = encoder_states.get_shape()[0].value  # batch_size if this line fails, it's because the batch size isn't defined
        attn_size = encoder_states.get_shape()[2].value  # 2*hz  if this line fails, it's because the attention length isn't defined
        q_attn_size = query_states.get_shape()[2].value  # 2*hz
        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, 2)  # now is shape (batch_size, attn_len, 1, attn_size)
        query_states = tf.expand_dims(query_states, 2)
        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size
        q_attention_vec_size = q_attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1],"SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

        # Get the weight vectors v
        v = tf.get_variable("v", [attention_vec_size])

        # Get the weight matrix W_q and apply it to each encoder state to get (W_q q_i), the query features
        W_q = tf.get_variable("W_q", [1, 1, q_attn_size, q_attention_vec_size])
        query_features = tf.nn.conv2d(query_states, W_q, [1, 1, 1, 1],"SAME")  # shape (batch_size,q_attn_length,1,q_attention_vec_size)

        # Get the weight vectors v_q
        v_q = tf.get_variable("v_q", [q_attention_vec_size])

        def background_attention(decoder_state):
            with tf.variable_scope("background_attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)  pass through
                decoder_features = util.linear(decoder_state, attention_vec_size, True)  # shape (batch_size, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)  # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_background_attention(e):
                    """Take softmax of e then apply enc_padding_mask"""
                    attn_dist = tf.nn.softmax(util.mask_softmax(enc_padding_mask, e))  # take softmax. shape (batch_size, attn_length)
                    return attn_dist

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])  # calculate e
                # Calculate attention distribution
                attn_dist = masked_background_attention(e) # batch_size,attn_length

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
                context_vector = tf.reshape(context_vector, [-1, attn_size])
            return context_vector, attn_dist

        def context_attention(decoder_state):
            with tf.variable_scope("context_attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                decoder_features = util.linear(decoder_state, q_attention_vec_size, True) # shape (batch_size, q_attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),1) # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_context_attention(e):
                    """Take softmax of e then apply enc_padding_mask"""
                    attn_dist = tf.nn.softmax(util.mask_softmax(que_padding_mask, e))  # take softmax. shape (batch_size, attn_length)
                    return attn_dist

                # Calculate v^T tanh(W_q q_i + W_s s_t + b_attn)
                f = tf.reduce_sum(v_q * tf.tanh(query_features + decoder_features), [2, 3])

                # Calculate attention distribution
                q_attn_dist = masked_context_attention(f)

                # Calculate the context vector from attn_dist and encoder_states
                q_context_vector = tf.reduce_sum(tf.reshape(q_attn_dist, [batch_size, -1, 1, 1]) * query_states, [1, 2])  # shape (batch_size, attn_size).
                q_context_vector = tf.reshape(q_context_vector, [-1, q_attn_size])

            return q_context_vector, q_attn_dist

        outputs = []
        background_attn_dists = []

        switcher_gen_pred_time_step = []
        switcher_gen_copy_time_step = []
        switcher_ref_time_step = []
        switcher_gen_time_step = []

        state = initial_state
        context_vector = tf.zeros([batch_size, attn_size])
        context_vector.set_shape([None, attn_size])
        q_context_vector = tf.zeros([batch_size, q_attn_size])
        q_context_vector.set_shape([None, q_attn_size])

        if initial_state_attention:  # true in decode mode
            context_vector, _ = background_attention(initial_state)
            q_context_vector, _ = context_attention(initial_state)

        for i, inp in enumerate(decoder_inputs):
            tf.logging.info("Adding hybrid_decoder timestep %i of %i", i + 1, len(decoder_inputs))
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = util.linear([inp] + [context_vector] + [q_context_vector], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):  # you need this because you've already run the initial attention(...) call
                    context_vector, attn_dist = background_attention(state)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):  # you need this because you've already run the initial attention(...) call
                    q_context_vector, q_attn_dist = context_attention(state)
            else:
                context_vector, attn_dist = background_attention(state)
                q_context_vector, q_attn_dist = context_attention(state)

            background_attn_dists.append(attn_dist)

            # Calculate  switcher
            with tf.variable_scope('calculate_switcher'):
                switcher_matrix = util.linear([context_vector, q_context_vector, state.c, state.h, x], 3, True)
                switcher_matrix = tf.nn.softmax(switcher_matrix)

                switcher_gen_pred_prob = tf.expand_dims(switcher_matrix[:, 0], 1)  # batch*1
                switcher_gen_copy_prob = tf.expand_dims(switcher_matrix[:, 1], 1)  # batch*1
                switcher_gen_prob = switcher_gen_pred_prob + switcher_gen_copy_prob  # batch*1
                switcher_ref_prob = tf.expand_dims(switcher_matrix[:, 2], 1)  # batch*1

                switcher_gen_pred_time_step.append(switcher_gen_pred_prob)
                switcher_gen_copy_time_step.append(switcher_gen_copy_prob)
                switcher_gen_time_step.append(switcher_gen_prob)
                switcher_ref_time_step.append(switcher_ref_prob)

            with tf.variable_scope("AttnOutputProjection"):
                output = util.linear([cell_output] + [context_vector] + [q_context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, background_attn_dists, switcher_ref_time_step, switcher_gen_time_step, switcher_gen_pred_time_step, switcher_gen_copy_time_step




