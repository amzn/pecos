# Session-Aware Query-Autocompletion using eXtreme Multi-Label Ranking, KDD 2021 

This folder contains code to train session-aware query-autocompletion models and reproduce experiments 
in ["Session-Aware Query-Autocompletion using eXtreme Multi-Label Ranking, KDD 2021"](https://arxiv.org/abs/2012.07654).

## Getting Started
* Clone the repository and enter `examples/qp2q` directory. 
* First create a [virtual environment](https://docs.python.org/3/library/venv.html) and then install dependencies 
by running the following command:
```bash 
pip install -r requirements.txt  

# NOTE: The original nltk version used in the experiment
# CAUTION: nltk<=3.6.3 is known to contain an Inefficient Regular Expression and is vulnerable to regular expression denial of service attacks
# Details: https://github.com/advisories/GHSA-2ww3-fxvq-293j
pip install nltk==3.4.5
```
If you're unfamiliar with Python virtual environments, check out the 
[user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

## Pre-procesing data
* Download AOL Search Logs data from [here](http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/).
* Extract files into `RAW_DATA_DIR` folder. This folder should contain 10 files with name template: ` user-ct-test-collection-**.txt`.
* Run the following command to pre-process the data and to create train/test/dev splits (as used in this paper).
    - ```bash 
      cd examples/qp2q 
      python utils/process_aol_dataset_from_orig_aol_files.py --data_dir <RAW_DATA_DIR> --out_dir <DATA_DIR> 
      ```
* The processed data is present in `DATA_DIR` folder, which contains three folders: `train`, `test`, and `dev`.
Each folder contains a `.json` file. Each line in the file contains a data point stored in JSON format.
  

## Training Models

* First run the setup script in `qp2q/bin` folder
    - ```bash 
        cd examples/qp2q
        source bin/setup.sh 
      ```
* Train a model
    - ```python models/train_model.py --config <path/to/config/file>  ```
    - Config files for various configurations of the models from [Table 2](https://arxiv.org/abs/2012.07654) in the paper 
      are present in `qp2q/config` folder. 
    - Please configure `fdir` parameter in config files to point to directory containing training data. By default,
    training data is assumed to present in `data/aol/train` directory where `data` directory is assumed to be present 
      in the same folder as `qp2q` folder.
    - Trained models will be stored in folders under `results` folder.
    - The `data`, `results` and `qp2q` folders are assumed to be present in the same directory, with the data 
      directory organized as illustrated below.
    ```
    ├── qp2q
    ├── results
    ├── data
    |    ├── aol
    |        ├── train
    |           ├── train.json
    |        ├── test
    |           ├── test.json
    |        ├── val
    |           ├── val.json
    ```


## Evaluation

* To generate query suggestions and evaluate the predictions for proposed models: 
    -  ```bash
       python eval/run_eval.py --gt <path/to/gt/file> --out_dir <path/to/result/dir> --model_dir <path/to/trained_model/folder>
       ```
* To generate query suggestions and evaluate the predictions for Most-Frequent-Query (MFQ) baseline:
    - First generate a dictionary mapping each prefix to a list of top-k (k=10) query suggestions and then run eval.  
      ```bash 
      python utils/create_pref_to_top_k_suggestions_dict.py --k 10 --fdir <train/data/folder> --out_fname </output/filename>
      python eval/run_mfq_eval.py --gt <path/to/gt/file> --out_dir <path/to/result/dir> --topk_file <path/to/prefix/to/topk/query/file> 
      ```
  
  
## Citation

If you find the code useful, please consider citing our paper.

* [Session-Aware Query Auto-completion using Extreme Multi-label Ranking (Yadav et al., KDD 2021)](https://arxiv.org/pdf/2012.07654.pdf)  [[bib]](../../bibtex/yadav2021session.bib)

```bibtex
@inproceedings{Yadav2021session,
author = {Yadav, Nishant and Sen, Rajat and Hill, Daniel N. and Mazumdar, Arya and Dhillon, Inderjit S.},
title = {Session-Aware Query Auto-Completion Using Extreme Multi-Label Ranking},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467087},
doi = {10.1145/3447548.3467087},
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
pages = {3835–3844},
numpages = {10},
keywords = {session-aware, extreme multi-label ranking, multi-label, auto-complete},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```

