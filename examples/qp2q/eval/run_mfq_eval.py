import os
import sys
import logging
import argparse

from qp2q.eval.gen_pred_data import generate_predictions
from qp2q.eval.eval_pred_latency import eval_pred_latency

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def eval_helper(topk_file, gt_file, out_dir, gen_pred_data, eval_latency):

    LOGGER.info("Param: GT_FILE={}".format(gt_file))
    LOGGER.info("Param: TOPK_FILE={}".format(topk_file))
    LOGGER.info("Param: GEN_PRED_DATA={}".format(gen_pred_data))
    LOGGER.info("Param: EVAL_LATENCY={}".format(eval_latency))

    gt_filename = gt_file.split("/")[-1]
    pred_data_path = f"{out_dir}/{gt_filename}.pred_data"

    # create eval config for evaluating with Most-Frequent-Query Baseline
    config_dict = {
        "name": "MFQ",
        "driver": "PrefFreqSuggester",
        "args": {
            "model_path": topk_file,
            "topk": 10,
        },
    }

    if gen_pred_data:
        generate_predictions(
            config_dict=config_dict, pred_data_path=pred_data_path, gt_data_file=gt_file
        )

    LOGGER.info("\n Computing eval metrics ")
    command = f"python eval/eval_pred_data.py  --out_file {out_dir}/eval_results.json --pred_file {pred_data_path} --gt_file {gt_file} "
    LOGGER.info(command)
    os.system(command)

    if eval_latency:
        LOGGER.info("Evaluating latency ")
        eval_pred_latency(
            config_dict=config_dict,
            gt_data_file=gt_file,
            pred_data_path=pred_data_path,
            num_samples=100000,
            warmup_samples=1000,
        )


def main():
    parser = argparse.ArgumentParser(description="Run evaluation using a  trained model \n ")
    parser.add_argument(
        "--topk_file",
        required=True,
        type=str,
        help="Dictionary file mapping each prefix to top-k most frequency queries matching the prefix",
    )
    parser.add_argument("--gt", required=True, type=str, help="Path to file with ground-truth data")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to save eval results")
    parser.add_argument(
        "--gen_pred_data", default=1, type=int, help="Whether to generate pred data or not"
    )
    parser.add_argument(
        "--eval_latency",
        default=0,
        type=int,
        help="Whether to evaluate latency for generation or not",
    )

    args = parser.parse_args()
    topk_file = args.topk_file
    out_dir = args.out_dir
    gt_file = args.gt
    gen_pred_data = bool(args.gen_pred_data)
    eval_latency = bool(args.eval_latency)

    LOGGER.info("Beginning eval")

    if os.path.isfile(gt_file):
        eval_helper(
            topk_file=topk_file,
            gt_file=gt_file,
            gen_pred_data=gen_pred_data,
            eval_latency=eval_latency,
            out_dir=out_dir,
        )
    else:
        LOGGER.info("Invalid gt file = {}".format(gt_file))


if __name__ == "__main__":
    main()
