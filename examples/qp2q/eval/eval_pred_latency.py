import os
import sys
import time
import json
import random
import logging
import argparse
import numpy as np
import os.path as path

from qp2q.eval.gen_pred_data import get_model

logger = logging.getLogger(__name__)


def _get_inference_time(model, sample):
    """Computes inference time for one call to get_suggestions."""
    start = time.time()
    model.get_suggestions(sample)
    end = time.time()
    return end - start


def _benchmark_model(model, gt_data_file, latency_data_file, num_samples, warmup_samples):
    """Benchmark latencies for a given model.

    Parameters:
    ----------
    model: obj
        model file or suggester
    gt_data_file: str
        path to ground truth data file
    latency_data_file: str
        path where the results are saved
    num_samples: int
        number of samples on which latency is evaluated.
    warmup_samples: int
        number of samples to be used for warmup.
    """
    logger.info("Reading groundtruth file.")
    with open(gt_data_file, "r") as fp:
        input_data = [json.loads(line) for line in fp]
    samples = random.sample(input_data, num_samples + warmup_samples)
    result = [_get_inference_time(model, sample) * 1000 for sample in samples]
    result = result[warmup_samples:]
    latency_dict = {
        "min": np.min(result),
        "max": np.max(result),
        "mean": np.mean(result),
        "p50": np.median(result),
        "p75": np.percentile(result, q=75),
        "p99": np.percentile(result, q=99),
    }
    logger.info("==== latency results in milli-seconds====")
    logger.info(latency_dict)
    with open(latency_data_file, "w") as fp:
        json.dump(latency_dict, fp)
    logger.info("Saved at {}".format(latency_data_file))


def eval_pred_latency(config_dict, pred_data_path, gt_data_file, num_samples, warmup_samples):
    """
    This method takes in model configurations to generate prediction results
    for given sample data.
    Latency results are then saved under save_dir
    * <save_dir>/<model_name>._latency_data.json
      * contains lists of query suggestions
    The model configs will be dumped into the following file:
    * <save_dir>/<model_name>_latency_config.json

    Parameters
    ----------
    config_dict : dict
                This dict contains the model and inference specific params
    pred_data_path: str
        Path to result file
    gt_data_file : str
        Ground-truth data file
    num_samples: int
        Number of samples to use for evaluating latency
    warmup_samples: int
        Number of samples to be used for warmup and not in latency calculations",
    """
    save_dir = os.path.dirname(pred_data_path)
    os.makedirs(path.abspath((save_dir)), exist_ok=True)

    # loading models
    logger.info("Loading models")
    model = get_model(config_dict)
    if not model:
        raise Exception("No models could be loaded")

    # save the eval configs (if not empty) for future reference
    _out = path.join(path.abspath(save_dir), config_dict["name"] + "_latency_config.json")
    json.dump(config_dict, open(_out, "w"))

    latency_data_file = pred_data_path.replace("pred_data", "latency_data")
    _benchmark_model(
        model=model,
        gt_data_file=gt_data_file,
        latency_data_file=latency_data_file,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
    )

    logger.info("Latency Eval finished")


def main(argv):

    parser = argparse.ArgumentParser("Evaluate inference latency of models")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path of the config json to compute predictions",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save prediction data"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to datafile containing ground-truth data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=10000,
        help="Number of samples to be used for evaluating latency",
    )
    parser.add_argument(
        "--warmup_samples",
        type=int,
        required=False,
        default=100,
        help="Number of samples to be used for warmup and not in latency calculations",
    )

    args = parser.parse_args(argv)
    config_file = args.config_file
    save_dir = args.save_dir
    gt_file = args.gt_file
    num_samples = args.num_samples
    warmup_samples = args.warmup_samples

    with open(config_file, "r") as f:
        config_dict = json.load(f)
    eval_pred_latency(
        config_dict=config_dict,
        save_dir=save_dir,
        gt_data_file=gt_file,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
    )


if __name__ == "__main__":
    logger.info(sys.argv)
    main(sys.argv)
