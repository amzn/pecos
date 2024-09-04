import json
import argparse
import os
from datasets import load_dataset
from pecos.xmr.reranker.data_utils import RankingDataUtils
from pecos.xmr.reranker.model import RankingModel
from tqdm import tqdm
import pandas as pd


def main(config_json_path: str):
    with open(config_json_path, "r") as fin:
        params = json.load(fin)

    params = RankingModel.EvalParams.from_dict(params)

    # helper function for getting the list of filepaths in a folder
    def construct_file_list(folder):
        return [os.path.join(folder, x) for x in os.listdir(folder)]

    input_files = construct_file_list(params.input_data_folder)
    label_files = construct_file_list(params.label_data_folder)
    target_files = construct_file_list(params.target_data_folder)
    inp_id_col = params.inp_id_col
    lbl_id_col = params.lbl_id_col

    input_files, label_files = RankingDataUtils.get_sorted_data_files(
        input_files, inp_id_col
    ), RankingDataUtils.get_sorted_data_files(label_files, lbl_id_col)

    table_stores = {
        "input": load_dataset("parquet", data_files=input_files, split="train"),
        "label": load_dataset("parquet", data_files=label_files, split="train"),
    }

    # Create output folder if it does not exist
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    outer_model = RankingModel.load(params.model_path)
    inner_model = outer_model.encoder
    if params.bf16:
        inner_model = inner_model.bfloat16()

    tokenizer = outer_model.tokenizer

    for target_file in tqdm(target_files):
        target_filename = os.path.basename(target_file)
        target_shucked_filename = ".".join(target_filename.split(".")[:-1])
        out_pre = params.output_file_prefix
        out_suff = params.output_file_suffix
        ext = ".parquet"
        save_filename = out_pre + target_shucked_filename + out_suff + ext

        eval_dataset = load_dataset(
            "parquet", data_files=[target_file], streaming=True, split="train"
        )

        results = RankingModel.predict(
            eval_dataset,
            table_stores,
            params,
            encoder=inner_model,
            tokenizer=tokenizer,
        )

        # Save the results to a parquet with (inp_id, lbl_id, score) columns
        # `results` is a list of tuple (inp_id, lbl_id, score)
        df = pd.DataFrame(results, columns=["inp_id", "lbl_id", "score"])
        df.to_parquet(os.path.join(params.output_dir, save_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config_json_path)
