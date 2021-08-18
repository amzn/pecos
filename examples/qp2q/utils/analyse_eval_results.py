import os
import csv
import sys
import json
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pygtrie import CharTrie
from collections import defaultdict
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def calc_mrr(gt_data_iterator, pred_data_iterator):
    all_mrr = []
    all_bleu = []
    for j, (gt_label, preds) in tqdm(
        enumerate(zip(gt_data_iterator, pred_data_iterator)), total=len(gt_data_iterator)
    ):
        preds = [p for p in json.loads(preds.lower())]
        mrr = 0.0
        for i, curr_pred in enumerate(preds):
            if gt_label == curr_pred:
                mrr = 1.0 / (i + 1)
                break

        all_mrr.append(mrr)
        wgtd_bleu_score = 0.0
        normalizer = 0.0
        for i, curr_pred in enumerate(preds):
            wgtd_bleu_score += (
                bleu_score.sentence_bleu(
                    [gt_label.split()],
                    curr_pred.split(),
                    smoothing_function=SmoothingFunction().method1,
                )
                / (i + 1)
            )
            normalizer += 1.0 / (i + 1)
        wgtd_bleu_score = wgtd_bleu_score / normalizer if normalizer > 0 else wgtd_bleu_score
        all_bleu.append(wgtd_bleu_score)

    return {
        "mrr": np.mean(all_mrr) if len(all_mrr) > 0 else 0.0,
        "mrr_err": np.std(all_mrr) if len(all_mrr) > 0 else 0.0,
        "bleu": np.mean(all_bleu) if len(all_bleu) > 0 else 0.0,
        "bleu_err": np.std(all_bleu) if len(all_bleu) > 0 else 0.0,
        "num_points": len(all_mrr),
    }


def get_data_by_freq_and_pref_len(gt_filename, pred_filename, label_trie, only_seen_labels):

    freq_to_data = defaultdict(list)
    pref_len_to_data = defaultdict(list)
    with open(gt_filename, "r") as all_gt_file, open(pred_filename, "r") as main_pred_file:
        for line_ctr, (line_gt, line_pred) in enumerate(zip(all_gt_file, main_pred_file)):
            gt_dict = json.loads(line_gt)
            gt_label = gt_dict["next_query"].lower()  # ground truth next_query
            prefix = gt_dict["prefix"].lower()  # prefix

            curr_label_freq = label_trie[gt_label] if gt_label in label_trie else 0

            freq_to_data[curr_label_freq] += [(gt_label, line_pred)]
            if only_seen_labels:
                if curr_label_freq > 0:
                    pref_len_to_data[len(prefix)] += [(gt_label, line_pred)]
                else:
                    pass
            else:
                pref_len_to_data[len(prefix)] += [(gt_label, line_pred)]

    new_freq_to_data = defaultdict(lambda: ([], []))
    for freq in freq_to_data:
        gt_labels, preds = zip(*freq_to_data[freq])
        new_freq_to_data[freq] = list(gt_labels), list(preds)

    new_pref_len_to_data = defaultdict(lambda: ([], []))
    for pref_len in pref_len_to_data:
        gt_labels, preds = zip(*pref_len_to_data[pref_len])
        new_pref_len_to_data[pref_len] = list(gt_labels), list(preds)

    return new_freq_to_data, new_pref_len_to_data


def write_results(pred_fname, gt_fname, feat_name, feat_to_scores, res_file):
    eval_metrics = list(calc_mrr([], []).keys())
    fieldnames = [feat_name] + eval_metrics + ["pred_fname", "gt_fname"]

    with open(res_file, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for feat in sorted(feat_to_scores):
            temp_dict = {feat_name: feat, "pred_fname": pred_fname, "gt_fname": gt_fname}
            temp_dict.update(feat_to_scores[feat])
            csv_writer.writerow(temp_dict)


def get_bucketed_scores(feat_to_data, feat_buckets):
    feat_bkt_to_scores = {}
    for min_feat_val, max_feat_val in zip(feat_buckets[:-1], feat_buckets[1:]):
        gt_data = []
        pred_data = []
        for curr_feat_val in range(min_feat_val, max_feat_val):
            gt_data += feat_to_data[curr_feat_val][0]
            pred_data += feat_to_data[curr_feat_val][1]

        feat_bkt_to_scores[min_feat_val, max_feat_val] = calc_mrr(
            gt_data_iterator=gt_data, pred_data_iterator=pred_data
        )

    return feat_bkt_to_scores


def analyse(gt_filename, pred_filename, label_trie, only_seen_labels):

    freq_to_data, pref_len_to_data = get_data_by_freq_and_pref_len(
        gt_filename=gt_filename,
        pred_filename=pred_filename,
        label_trie=label_trie,
        only_seen_labels=only_seen_labels,
    )

    freq_to_scores = {
        freq: calc_mrr(gt_data_iterator=gt_data, pred_data_iterator=pred_data)
        for freq, (gt_data, pred_data) in freq_to_data.items()
    }

    pref_len_to_scores = {
        pref_len: calc_mrr(gt_data_iterator=gt_data, pred_data_iterator=pred_data)
        for pref_len, (gt_data, pred_data) in pref_len_to_data.items()
    }

    out_dir = os.path.dirname(pred_filename)

    test_fname = gt_filename.split("/")[-1]
    # Write results for each label freq to csv file
    res_file = f"{out_dir}/{test_fname}_eval_res_by_seen_in_train_bkt_{only_seen_labels}.csv"
    bktd_freq_to_scores = get_bucketed_scores(feat_to_data=freq_to_data, feat_buckets=[0, 1, 80000])
    write_results(
        pred_fname=pred_filename,
        feat_name="freq_bkt",
        feat_to_scores=bktd_freq_to_scores,
        res_file=res_file,
        gt_fname=gt_filename,
    )

    res_file = f"{out_dir}/{test_fname}_eval_res_by_freq_{only_seen_labels}.csv"
    write_results(
        pred_fname=pred_filename,
        feat_name="freq",
        feat_to_scores=freq_to_scores,
        res_file=res_file,
        gt_fname=gt_filename,
    )

    res_file = f"{out_dir}/{test_fname}_eval_res_by_freq_bkt_{only_seen_labels}.csv"
    bktd_freq_to_scores = get_bucketed_scores(
        feat_to_data=freq_to_data, feat_buckets=[0, 1, 32, 128, 512, 2048, 8192, 32000, 80000]
    )
    write_results(
        pred_fname=pred_filename,
        feat_name="freq_bkt",
        feat_to_scores=bktd_freq_to_scores,
        res_file=res_file,
        gt_fname=gt_filename,
    )

    # Write results for each label prefix-length to csv file
    res_file = f"{out_dir}/{test_fname}_eval_res_by_pref_len_{only_seen_labels}.csv"
    write_results(
        pred_fname=pred_filename,
        feat_name="pref_len",
        feat_to_scores=pref_len_to_scores,
        res_file=res_file,
        gt_fname=gt_filename,
    )

    # Write results for each label prefix-length to csv file bucketed
    max_pref_bkt = 26
    max_pref = max(pref_len_to_scores.keys())
    pref_feat_bkts = list(range(max_pref_bkt))
    if max_pref > max_pref_bkt:
        pref_feat_bkts = pref_feat_bkts + [max_pref + 1]
    res_file = (
        f"{out_dir}/{test_fname}_eval_res_by_pref_len_bkt_{max_pref_bkt}_{only_seen_labels}.csv"
    )
    bktd_pref_len_to_scores = get_bucketed_scores(
        feat_to_data=pref_len_to_data, feat_buckets=pref_feat_bkts
    )
    write_results(
        pred_fname=pred_filename,
        feat_name="pref_len_bkt",
        feat_to_scores=bktd_pref_len_to_scores,
        res_file=res_file,
        gt_fname=gt_filename,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Breakdown performance of model given prediction file"
    )
    parser.add_argument(
        "--gt_file", type=str, required=True, help="file containing ground-truth data"
    )
    parser.add_argument(
        "--pred_file", type=str, required=True, help="file containing prediction data"
    )
    parser.add_argument(
        "--label_file", type=str, default=None, help="file containing label freq trie"
    )

    np.random.seed(0)

    args = parser.parse_args()
    _gt_file = args.gt_file
    _pred_file = args.pred_file
    _label_file = args.label_file if args.label_file != "" else None

    LOGGER.info("Loading label freq trie from {}".format(_label_file))
    if _label_file is None:
        label_trie = defaultdict(int)
    else:
        with open(_label_file, "rb") as f:
            label_trie = pickle.load(f)
        assert isinstance(label_trie, CharTrie)
    LOGGER.info("Loaded label freq trie")

    analyse(gt_filename=_gt_file, pred_filename=_pred_file, label_trie=label_trie)


def final_paper_analysis():
    parser = argparse.ArgumentParser(
        description="Breakdown performance of model given prediction file"
    )
    parser.add_argument("--opt", type=int, default=-1, help="option")
    parser.add_argument(
        "--base_dir", type=str, required=True, help="dir with all relevant result folder"
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default="",
        help="pickle file containing dict/trie that maps " "label to its frequency",
    )
    parser.add_argument("--gt_file", type=str, required=True, help="json file containing gt data")
    args = parser.parse_args()
    opt = args.opt
    base_dir = args.base_dir
    label_file = args.label_file
    gt_file = args.gt_file

    pecos_08 = f"{base_dir}/7_ReproduceRes/d=aol_v=poswgtd_c-tfidf_i=hybridindexer_s=0_08"
    freq = f"{base_dir}/8_FinalResults/PrefFreqSuggester"
    seq2seqGRU = f"{base_dir}/8_Baseline/d=aol_m=HRED_s=0"
    seq2seqGRU_freq = f"{base_dir}/8_Baseline/d=aol_m=HRED_s=0_rerank"

    LOGGER.info("Loading label freq trie from {}".format(label_file))
    if label_file is None:
        label_trie = defaultdict(int)
        only_seen_labels = False
    else:
        label_trie = pickle.load(open(label_file, "rb"))
        only_seen_labels = True
        assert isinstance(label_trie, CharTrie)
    LOGGER.info("Loaded label freq trie")

    if opt == 1:
        all_dirs = [pecos_08]
    elif opt == 2:
        all_dirs = [freq]
    elif opt == 3:
        all_dirs = [seq2seqGRU]
    elif opt == 4:
        all_dirs = [seq2seqGRU_freq]
    else:
        all_dirs = [pecos_08, freq, seq2seqGRU, seq2seqGRU_freq]

    for curr_dir in tqdm(all_dirs):
        print("Running for dir = {}".format(curr_dir))
        try:
            pred_file = f"{curr_dir}/eval/{curr_dir.split('/')[-1]}_test.json.pred_data"
            analyse(
                gt_filename=gt_file,
                pred_filename=pred_file,
                label_trie=label_trie,
                only_seen_labels=only_seen_labels,
            )
            analyse(
                gt_filename=gt_file,
                pred_filename=pred_file,
                label_trie=defaultdict(int),
                only_seen_labels=False,
            )
        except Exception as e:
            print(f"1) Error processing {curr_dir} {e}")


if __name__ == "__main__":
    # main()
    final_paper_analysis()
