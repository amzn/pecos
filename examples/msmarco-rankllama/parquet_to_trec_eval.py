
import argparse
import os
import pandas as pd


def main(args):
    """
    Combine all results from the results folder and write them to the output file.
    """
    result_files = [
        os.path.join(args.input_parquet_path, x)
        for x in os.listdir(args.input_parquet_path)
    ]
    all_results = pd.read_parquet(result_files[0])
    for f in result_files[1:]:
        all_results = pd.concat([all_results, pd.read_parquet(f)])
    # sort all results by 'inp_id' and then 'score' in descending order    
    all_results = all_results.sort_values(by=['inp_id', 'score'], ascending=[True, False])

    cur_inp_id = None
    with open(args.output_trec_path, "w") as fout:
        for row in all_results.itertuples():
            if cur_inp_id != row.inp_id:
                cur_inp_id = row.inp_id
                rank = 0
            rank += 1
            fout.write(f"{row.inp_id} Q0 {row.lbl_id} {rank} {row.score} dense\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-parquet-path", type=str, required=True)
    parser.add_argument("-o", "--output-trec-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
