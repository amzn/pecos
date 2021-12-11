## Preliminaries: Install all the required packages.

To install all dependencies, run the following command:
```
pip install -r requirements.txt
```


## Prepare the dataset.


* Download datasets following instructions in ``dataset/README.md``.  
* The following files should be available in <work_dir>/dataset/<dataset>:
    - trn.json
    - tst.json
    - lbl.json
    - all_pairs.txt


## Configure the distributed training of Accelerate

Accelerate provides a CLI tool that unifies all launcher. To use it, just run
```bash
accelerate config
```
on your machine and reply to the questions asked.

Or you can directly use the config file ``accelerate_config.yaml`` when you run the script.

## Stage I: Multi-scale Adaptive Clustering and Label Regularization

To run the pre-training Stage I, here is an example on LF-Amazon-131K:
```bash
export dataset=LF-Amazon-131K
export mode=ict
export log=test
export model=bert-base-uncased

bash run.sh $dataset $mode $log $model
```

## Generate pseudo positive pairs using the encoder
You can run the following script to generate pseudo positive pairs from the encoder for self training
```bash
export dataset=LF-Amazon-131K
export mode=construct-pseudo
export log=test
export model=<your model path>
bash evaluate.sh $dataset $mode $log $model
```

Then you will have a file containing top 5 potential labels for each training instance at ``<work_dir>/dataset/<dataset>/pseudo_pos.json``.

## Stage II: Self-training with pseudo positive pairs
For Stage II, you only need to change ``mode`` to ``self-train`` and provide the path to the current model
```bash
export dataset=LF-Amazon-131K
export mode=self-train
export log=test
export model=<your model path>

bash run.sh $dataset $mode $log $model
```

## Fine-tune the encoder on few-shot data sampled by label
First, prepare ``label_index.json`` for sampling the subset by label
```bash
python label_index.py --dataset LF-Amazon-131K
```

Then change ``mode`` to ``finetune-label`` to start the fine-tuning procedure
```bash
export dataset=LF-Amazon-131K
export mode=finetune-label
export log=test
export model=<your model path>

bash run.sh $dataset $mode $log $model
```

## Fine-tune the encoder on few-shot data sampled by pair

Change ``mode`` to ``finetune-pair`` to start the fine-tuning procedure
```bash
export dataset=LF-Amazon-131K
export mode=finetune-pair
export log=test
export model=<your model path>

bash run.sh $dataset $mode $log $model
```

## Evaluate the pre-trained model
We also provide pre-trained models for all four datasets at [this link](https://archive.org/download/maclr-www22/pretrained-models/). You can download them with ``wget``, decompress the model, and run the following command to evaluate the model
```bash
export dataset=LF-Amazon-131K
export mode=evaluate
export log=test
export model=<your model path>

bash evaluate.sh $dataset $mode $log $model
```