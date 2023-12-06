# Experiment Code for PEFA, WSDM 2024

This folder contains code to reproduce experiments in
["PEFA: Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models"](https://arxiv.org/abs/2312.02429)

## 1. Summary
In this repository, we demonstrated how to reproduce Table 2 (NQ-320K) and Table 3 (Trivia-QA) of our PEFA paper.
After following Steps 2-6 in the subsequent sections, you should be able to obtain

| NQ-320K | Recall@10 | Recall@100 |
|---|---|---|
| DistilBERT + PEFA-XS |  80.52% | 92.23% |
| DistilBERT + PEFA-XL |  85.26% | 92.53% |
| MPNet + PEFA-XS |  86.67% | 94.53% |
| MPNet + PEFA-XL |  88.72% | 95.13% |
| Sentence-T5-base + PEFA-XS |  82.52% | 92.18% |
| Sentence-T5-base + PEFA-XL |  83.69% | 92.55% |
| GTR-T5-base + PEFA-XS |  84.90% | 93.28% |
| GTR-T5-base + PEFA-XL |  88.71% | 94.36% |

| Trivia-QA | Recall@20 | Recall@100 |
|---|---|---|
| DistilBERT + PEFA-XS |  86.28% | 93.33% |
| DistilBERT + PEFA-XL |  84.18% | 91.24% |
| MPNet + PEFA-XS |  86.05% | 92.97% |
| MPNet + PEFA-XL |  86.13% | 92.42% |
| Sentence-T5-base + PEFA-XS |  78.39% | 88.57% |
| Sentence-T5-base + PEFA-XL |  75.13% | 87.24% |
| GTR-T5-base + PEFA-XS |  83.81% | 91.02% |
| GTR-T5-base + PEFA-XL |  85.30% | 92.38% |


## 2. Getting Started
* Clone the repository and enter `examples/pefa-wsdm24` directory.
* First create a [virtual environment](https://docs.python.org/3/library/venv.html) and then install dependencies by running the following command:
```bash
python3 -m pip install libpecos==1.2.1
python3 -m pip install sentence-transformers==2.2.1
``` 

## 3. Download Pre-processed Data for NQ320K and Trivia-QA
Our pre-processed datasets of NQ320K and Trivia-QA can be download at
```bash
mkdir -p ./data/xmc; cd ./data/xmc;
DATASET="nq320k"  # nq320k or trivia
wget https://archive.org/download/pefa-wsdm24/data/xmc/${DATASET}.tar.gz
tar -zxvf ./${DATASET}.tar.gz
cd ../../  # get back to the pecos/examples/pefa-wsdm24 directory
```

Additional Notes on data-preprocessing
* We first obtained original NQ320K/Trivia-QA datasets from the [NCI Paper, Wang et al., NeurIPS22](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)
- We then pre-processed it into our format.
- Details about our data pre-processing scripts can be found in `./data/README.md`.


## 4. Generate Embeddings for PEFA Inference
Before running PEFA inference, we select an encoder to generating query/passage embeddings
```bash
DATASET="nq320k"  # nq320k or trivia
ENCODER="gtr-t5-base"
bash run_encoder ${DATASET} ${ENCODER}
```
The embeddings will be saved to `./data/embeddings/${DATSET}/`
Regarding the `ENCODER` used in our paper,
* For `nq320k`, we consider `{nq-distilbert-base-v1, multi-qa-mpnet-base-dot-v1, sentence-t5-base, gtr-t5-base}`
* For `trivia`, we consider `{multi-qa-distilbert-dot-v1, multi-qa-mpnet-base-dot-v1, sentence-t5-base, gtr-t5-base}`

## 5. Run PEFA-XS
```bash
DATASET="nq320k"
ENCODER="gtr-t5-base"
bash run_pefa_xs.sh ${DATASET} ${ENCODER}
```
The script `run_pefa_xs.sh` calls the `pefa_xs.py` with hard-coded hyper-parameters.
For example, it uses `threads=64`. If your machine has less CPU cores, please adjust it accordingly.

## 6. Run PEFA-XL
```bash
DATASET="nq320k"
ENCODER="gtr-t5-base"
bash run_pefa_xl.sh ${DATASET} ${ENCODER}
```
The script `run_pefa_xl.sh` calls the `pefa_xl.py` with hard-coded hyper-parameters.
For example, it uses `threads=64`. If your machine has less CPU cores, please adjust it accordingly.

## 7. Citation
If you find this work useful for your research, please cite:
```
@inproceedings{chang2024pefa,
  title={PEFA: Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models},
  author={Wei-Cheng Chang and Jyun-Yu Jiang and Jiong Zhang and Mutasem Al-Darabsah and Choon Hui Teo and Cho-Jui Hsieh and Hsiang-Fu Yu and S. V. N. Vishwanathan},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM '24)},
  year={2024}
}
```
