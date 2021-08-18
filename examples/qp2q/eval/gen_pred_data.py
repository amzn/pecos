import os
import sys
import json
import logging
import os.path as path
import argparse

from tqdm import tqdm
from qp2q.eval.eval_helper import WorkerMeta

logger = logging.getLogger(__name__)


def get_model(eval_config):
    model_info = WorkerMeta.subtypes
    logger.info("Loading {0} for {1}".format(eval_config["driver"], eval_config["name"]))

    model_klass = model_info[eval_config["driver"]]
    model = model_klass(**eval_config["args"]) if "args" in eval_config else model_klass()

    return model


def generate_predictions(config_dict, pred_data_path, gt_data_file):
    """
    This method takes in model configurations to generate prediction results
    for given sample data. The prediction results are then saved under save_dir
    * <save_dir>/<model_name>.pred_data
      * contains lists of query suggestions
    The model configs will be dumped into the following file:
    * <save_dir>/<model_name>.json

    Parameters
    ----------
    config_dict : dict
                This dict contains the model and inference specific params
    pred_data_path: Name of output file
    gt_data_file : Ground-truth data
    """
    save_dir = os.path.dirname(pred_data_path)
    os.makedirs(path.abspath(save_dir), exist_ok=True)

    # loading samples
    logger.info("Loading samples")
    gt_data = []
    with open(gt_data_file) as reader:
        gt_data = [json.loads(line) for line in reader]

    # loading models
    logger.info("Loading models")
    model = get_model(config_dict)
    if not model:
        raise Exception("No models could be loaded")

    # save the eval configs (if not empty) for future reference
    _out = path.join(path.abspath(save_dir), config_dict["name"] + "_config.json")
    json.dump(config_dict, open(_out, "w"))

    logger.info("Beginning eval")
    with open(pred_data_path, "w") as pred_file:
        for curr_gt_data_item in tqdm(gt_data):
            # Getting query suggestions from models
            next_query_suggestions = model.get_suggestions(curr_gt_data_item)
            single_line = json.dumps(next_query_suggestions)
            pred_file.write(single_line + "\n")

    logger.info("Eval finished")


def main(argv):

    parser = argparse.ArgumentParser("Generate next query predictions")
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
    args = parser.parse_args(argv)
    config_file = args.config_file
    save_dir = args.save_dir
    gt_file = args.gt_file

    with open(config_file, "r") as f:
        config_dict = json.load(f)
    generate_predictions(config_dict=config_dict, save_dir=save_dir, gt_data_file=gt_file)


if __name__ == "__main__":
    logger.info(sys.argv)
    main(sys.argv)
