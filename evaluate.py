import sys
import glob
import json
import os
import time
from metrics import rouge, bleu, f1


def rounder(num):
    return round(num, 2)


def bleu_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = cal_bleu([prediction], [ground_truth])
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def rouge_max_over_ground_truths(prediction, ground_truths):
    scores_for_rouge1 = []
    scores_for_rouge2 = []
    scores_for_rougel = []
    for ground_truth in ground_truths:
        score = cal_rouge([prediction], [ground_truth])
        scores_for_rouge1.append(score[0])
        scores_for_rouge2.append(score[1])
        scores_for_rougel.append(score[2])
    return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)


def cal_bleu(infer, ref):

    while True:
        try:
            bleu_score = bleu.moses_multi_bleu(infer, ref)
            return bleu_score
        except FileNotFoundError:
            print("Failed to test bleu_score. Sleeping for %i secs...", 0.01)
            time.sleep(3)


def cal_rouge(infer, ref):
    x = rouge.rouge(infer, ref)
    return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100


def evaluate(infer, ref, inferred_spans, ref_spans):
    bl = cal_bleu(infer, ref)
    x = rouge.rouge(infer, ref)
    f, e, total = f1.evaluate(inferred_spans, ref_spans)
    return bl, x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100, f, e, total


def evaluate_multi_ref(inferred_response, inferred_spans, example_id):
    ref_spans = []
    ref_responses = []
    print("load multi reference data")
    with open("data/modified_multi_reference_test.json", 'r', encoding='utf-8') as r:
        multi_reference_test = json.load(r)
    assert len(multi_reference_test) == len(example_id), "the length of multi_ref example should be same as pre"
    for i in example_id:
        ref_spans.append(multi_reference_test[i]["spans"])
        ref_responses.append(multi_reference_test[i]["responses"])
    print("calculate f1 metric")
    # calculate f1 metric
    f, e, total_span = f1.evaluate(inferred_spans, ref_spans)
    # calculate bleu and rouge
    print("multi_f1:", f)
    print("multi_em:", e)
    print("span total:", total_span)

    print("calculate bleu and rouge")
    bleu = rouge_1 = rouge_2 = rouge_l = total = 0
    assert len(inferred_response) == len(ref_responses), "the length of predicted span and ground_truths span should be same"
    for i, pre in enumerate(inferred_response):
        print("calculating %d " % (i+1))
        bleu += bleu_max_over_ground_truths(pre, ref_responses[i])
        rouge_result = rouge_max_over_ground_truths(pre, ref_responses[i])
        rouge_1 += rouge_result[0]
        rouge_2 += rouge_result[1]
        rouge_l += rouge_result[2]
        total += 1

    bleu = bleu / total
    rouge_1 = rouge_1 / total
    rouge_2 = rouge_2 / total
    rouge_l = rouge_l / total

    return bleu, rouge_1, rouge_2, rouge_l, f, e, total_span


def main(model_path, log_root, decode_dir, mode, multi_label_eval=False):
    # statr evaluation
    with open(os.path.join(decode_dir, "output.json"), 'r', encoding='utf-8') as r:
        output = json.load(r)
    example_index = list(output.keys())
    ref_response = []
    inferred_response = []
    ref_spans = []
    inferred_spans = []
    gen_ref_num = 0

    for i in example_index:
        ref_response.append(output[i]["ref_response"])
        inferred_response.append(output[i]["inferred_response"])
        ref_spans.append([output[i]["ref_span"]])
        inferred_spans.append(output[i]["inferred_spans"])

        num_ref = False
        num_gen = False
        for item in output[i]["output_index"]:
            if isinstance(item, list):
                num_ref = True
            elif isinstance(item, int):
                num_gen = True
        if num_ref and num_gen:
            gen_ref_num = gen_ref_num+1

    assert len(inferred_response) == len(ref_response), "the length of infer_response and ref_responses should be same "

    print("start single reference evaluation")
    result = evaluate(inferred_response, ref_response, inferred_spans, ref_spans)

    try:
        with open(os.path.join(log_root, str(mode)+"_result.json"), 'r', encoding='utf-8') as r:
            result_log = json.load(r)

    except FileNotFoundError:
        with open(os.path.join(log_root, str(mode)+"_result.json"), 'w', encoding='utf-8') as w:
            result_log = {}
            json.dump(result_log, w)

    result_log[model_path] = {"bleu": rounder(float(result[0])), "rouge_1": rounder(float(result[1])), "rouge_2": rounder(float(result[2])), "rouge_l": rounder(float(result[3])), "f1": rounder(float(result[4])), "exact_match": rounder(float(result[5])), "span_num": result[6],"gen_ref_num":gen_ref_num}

    with open(os.path.join(log_root, str(mode)+"_result.json"), 'w', encoding='utf-8') as w:
        json.dump(result_log, w)

    print("finish single reference evaluation")

    if mode == "test" and multi_label_eval:
        print("start multi reference evaluation for test")
        multi_ref_result_log = {}
        multi_ref_result = evaluate_multi_ref(inferred_response, inferred_spans, example_index)
        multi_ref_result_log[model_path] = {"multi_ref_bleu": rounder(float(multi_ref_result[0])), "multi_ref_rouge_1": rounder(float(multi_ref_result[1])), "multi_ref_rouge_2": rounder(float(multi_ref_result[2])),
                                  "multi_ref_rouge_l": rounder(float(multi_ref_result[3])), "multi_ref_f1": rounder(float(multi_ref_result[4])), "multi_ref_exact_match": rounder(float(multi_ref_result[5])),
                                  "span_num": multi_ref_result[6],"gen_ref_num": gen_ref_num}

        with open(os.path.join(log_root, str(mode)+"_multi_result.json"), 'w', encoding='utf-8') as w:
            json.dump(multi_ref_result_log, w)
        print("all evaluation is finished")


if __name__ == '__main__':
    mode = "test"
    train_dir = "log/Camera_Ready_2_RefNet/train/"
    model_dir = "log/Camera_Ready_2_RefNet/train/model.ckpt-10775"

    main(model_dir, "log/Camera_Ready_2_RefNet", "log/Camera_Ready_2_RefNet/55_Test_Infer_ckpt-10775", mode, True)

    r = open(os.path.join(train_dir, "finished_"+mode+"_models.json"), 'r', encoding='utf-8')
    finished_option_models = json.load(r)
    r.close()

    finished_option_models["finished_"+mode+"_models"].append(model_dir)

    w = open(os.path.join(train_dir, "finished_"+mode+"_models.json"), 'w', encoding='utf-8')
    json.dump(finished_option_models, w)
    w.close()

