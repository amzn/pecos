# PECOS XMR Reranker on MS-Marco Dataset

This is an example of PECOS-based RankingModel that reproduced the [RankLlaMA paper](https://arxiv.org/abs/2310.08319).

## How to run

### Training
```bash
torchrun --nnodes 1 --nproc-per-node 8 \
    -m pecos.xmr.reranker.train \
    --config_json_path ./msmarco_qwen2-7B.train.json
```

### Predictions
```bash
python -m pecos.xmr.reranker.predict \
    --config_json_path ./msmarco_qwen2-7B.pred.json
```

## Evaluation
We first convert the predictions from parquet to TREC format:
```python
python -u parquet_to_trec_eval.py -i inference_outputs/ms_marco/qwen2-7B -o inference_outputs/ms_marco/qwen2-7B.pred.trec
```

We then follow [Pyserini]() evaluation protocol to eval the NDCG@10,
and you should see the results like:
```python
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage inference_outputs/ms_marco/qwen2-7B.pred.trec 

Results:
ndcg_cut_10             all     0.7619
```
