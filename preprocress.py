import os
import json
import struct
import collections
from tensorflow.core.example import example_pb2
import re
import spacy

nlp = spacy.load('en', disable=['tagger', 'ner'], vectors=False)
print('Spacy loaded')


def get_tokens(doc):
    doc = nlp(doc)
    new_tokens = []
    for k in doc:
        new_tokens.append(k.text)
    return new_tokens


# We use these to separate the summary sentences in the .bin datafiles
CHUNK_SIZE = 1000


def chunk_file(chunks_dir, finished_files_dir, set_name):
    in_file = finished_files_dir + '/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(chunks_dir, finished_files_dir):
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(chunks_dir, finished_files_dir, set_name)
    print("Saved chunked data in %s" % chunks_dir)


def write_to_bin(url_file, out_file, finished_files_dir, makevocab=False):
    url_list = url_file
    VOCAB_SIZE = 25000
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(url_list):
            if idx % 1000 == 0:
                print("Writing story %i  percent done" % idx)

            background, context, response, span, b_start, b_end, r_start, r_end, example_id = get_art_abs(s)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['background'].bytes_list.value.extend([background.encode()])
            tf_example.features.feature['context'].bytes_list.value.extend([context.encode()])
            tf_example.features.feature['response'].bytes_list.value.extend([response.encode()])
            tf_example.features.feature['span'].bytes_list.value.extend([span.encode()])
            tf_example.features.feature['b_start'].bytes_list.value.extend([b_start.encode()])
            tf_example.features.feature['b_end'].bytes_list.value.extend([b_end.encode()])
            tf_example.features.feature['r_start'].bytes_list.value.extend([r_start.encode()])
            tf_example.features.feature['r_end'].bytes_list.value.extend([r_end.encode()])
            tf_example.features.feature['example_id'].bytes_list.value.extend([example_id.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = background.split(' ')
                abs_tokens = response.split(' ')
                que_tokens = context.split(' ')

                tokens = art_tokens + abs_tokens + que_tokens
                tokens = [t.strip() for t in tokens]
                tokens = [t for t in tokens if t != ""]
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def get_art_abs(story_file):
    background = str(story_file['background'])
    context = str(story_file['context'])
    response = str(story_file['response'])
    span = str(story_file['span'])
    b_start = str(story_file['b_start'])
    b_end = str(story_file['b_end'])
    r_start = str(story_file['r_start'])
    r_end = str(story_file['r_end'])
    example_id = str(story_file['example_id'])

    re.sub("\s+", " ", background)
    re.sub("\s+", " ", context)
    re.sub("\s+", " ", response)
    re.sub("\s+", " ", span)

    return background, context, response, span, b_start, b_end, r_start, r_end, example_id


def process_tokens(st):
    return " ".join(st)


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def RefNet(data, query_type, data_type, start_type):
    folder_name = data_type + '_' + query_type
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  #

    finished_files_dir = folder_name + "/finished_files"
    chunks_dir = os.path.join(finished_files_dir, "chunked")
    train_data = []
    valid_data = []
    test_data = []
    
    background_span_error = 0
    background_span_error1 = 0
    response_span_error = 0
    
    for k, type_data in enumerate(data):
        for count, i in enumerate(type_data):
            if count % 1000 == 0:
                print(count)
            background = i[data_type].lower()
            context = i[query_type].lower()
            response = i['response'].lower()
            span = i['span'].lower().strip()

            background_span_char_start = background.find(span)
            response_spans_char_start = response.find(span)
            span_lenth = len(span)
            background_span_char_end = span_lenth + background_span_char_start
            response_spans_char_end = span_lenth + response_spans_char_start

            if (background_span_char_start != i[start_type]) and (i[start_type]!=None):
                background_span_error = background_span_error + 1
                print("No.{}, The matching answer_start is different with the author label".format(background_span_error))
                print("The author label：", i[start_type])
                print("The matching number：", background_span_char_start)
                print(i['example_id'])
                print("background: ", background)
                print("context: ", context)
                print("response: ", response)
                print("span: ", span)
                background_span_char_start = i[start_type]
                background_span_char_end = span_lenth + background_span_char_start
                print("Modify according to the the author label" % background_span_char_start)

            background_token = get_tokens(background)
            context_token = get_tokens(context)
            response_token = get_tokens(response)
            span_token = get_tokens(span)
            
            background_refine_text = process_tokens(background_token)
            context_token_refine_text = process_tokens(context_token)
            response_token_refine_text = process_tokens(response_token)
            span_token_refine_text = process_tokens(span_token)
                   
            background_token = background_refine_text.split()
            context_token = context_token_refine_text.split()
            response_token = response_token_refine_text.split()
            span_token = span_token_refine_text.split()

            background_spans = convert_idx(background, background_token)
            response_spans = convert_idx(response, response_token)

            background_span = []
            for idx, b_span in enumerate(background_spans):
                if not (background_span_char_end <= b_span[0] or background_span_char_start >= b_span[1]):
                    background_span.append(idx)
            b_start, b_end = background_span[0], background_span[-1]

            response_span = []
            for idx, r_span in enumerate(response_spans):
                if not (response_spans_char_end <= r_span[0] or response_spans_char_start >= r_span[1]):
                    response_span.append(idx)
            r_start, r_end = response_span[0], response_span[-1]

            if response_token[r_start:(r_end + 1)] != span_token:
                response_span_error = response_span_error + 1
                print("No.{}, The span extracted from response is different from the span labeled by author".format(response_span_error))
                print("The author label：", span_token)
                print("The span extracted from response：", response_token[r_start:(r_end + 1)])
                print(i['example_id'])
                print("background: ", background)
                print("context: ", context)
                print("response: ", response)
                print("span: ", span)

            if background_token[b_start:(b_end + 1)] != span_token:
                background_span_error1 = background_span_error1 + 1
                print("No.{}, The span extracted from background is different from the span labeled by author".format(background_span_error1))
                print("The author label：", span_token)
                print("The span extracted from response：", background_token[b_start:(b_end + 1)])
                print(i['example_id'])
                print("background: ", background)
                print("context: ", context)
                print("response: ", response)
                print("span: ", span)

            example = {'background': process_tokens(background_token),
                       'context': process_tokens(context_token),
                       'response': process_tokens(response_token),
                       'span': process_tokens(span_token),
                       'b_start': b_start,
                       'b_end': b_end,
                       'r_start': r_start,
                       'r_end': r_end,
                       'example_id': i['example_id']
                       }

            if k == 0:
                train_data.append(example)  # [{train_example1},{train_example2}...]
            elif k == 1:
                valid_data.append(example)
            else:
                test_data.append(example)

    all_train_urls = train_data
    all_val_urls = valid_data
    all_test_urls = test_data

    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"), finished_files_dir)
    write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"), finished_files_dir)
    write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), finished_files_dir,makevocab=True)

    # Chunk the data.
    # This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all(chunks_dir, finished_files_dir)

    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))


train_data = json.load(open('train_data.json', 'r'))
test_data = json.load(open('test_data.json', 'r'))
valid_data = json.load(open('dev_data.json', 'r'))
data = [train_data, valid_data, test_data]

RefNet(data, 'context', 'mixed', "answer_start_mixed")
