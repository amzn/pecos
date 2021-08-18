import argparse
import logging
import numpy as np
from pathlib import Path
import json
import sys
import os
import csv
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def calc_mrr_bleu(gt_data_iterator, pred_data_iterator):

    all_mrr = []
    all_bleu = []
    for j, (gt_line, preds) in tqdm(enumerate(zip(gt_data_iterator, pred_data_iterator))):
        gt_dict = json.loads(gt_line)
        gt_label = gt_dict["next_query"].lower()  # ground truth next_query
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

    return {"mrr": np.mean(all_mrr), "bleu": np.mean(all_bleu), "num_samples": len(all_mrr)}


def calc_mrr_bleu_from_file(gt_file, pred_file):
    with open(gt_file, "r") as orig_gt_reader, open(pred_file, "r") as pred_reader:
        return calc_mrr_bleu(gt_data_iterator=orig_gt_reader, pred_data_iterator=pred_reader)


def main():
    parser = argparse.ArgumentParser(description="Eval predictions given gt data")
    parser.add_argument("--gt_file", type=str, required=True, help="gt data file")
    parser.add_argument("--pred_file", type=str, required=True, help="pred data file")
    parser.add_argument("--out_file", type=str, required=True, help="File to save results in")

    args = parser.parse_args()
    _gt_file = args.gt_file
    _pred_file = args.pred_file
    _out_file = args.out_file

    out_dir = os.path.dirname(_out_file)
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    res = calc_mrr_bleu_from_file(gt_file=_gt_file, pred_file=_pred_file)
    LOGGER.info("Eval Result")
    LOGGER.info(json.dumps(res))

    res["gt_file"] = _gt_file
    res["pred_file"] = _pred_file

    _out_file = _out_file + ".csv"
    if os.path.exists(_out_file):
        with open(_out_file, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=res.keys())
            writer.writerow(res)
    else:
        with open(_out_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=res.keys())
            writer.writeheader()
            writer.writerow(res)


if __name__ == "__main__":
    main()
