# KDD 2022 Hands-on Tutorial - PECOS: Prediction for Enormous and Correlated Output Spaces

In this tutorial, we will introduce several key functions and features of the PECOS library.
By way of real-world examples, the attendees will learn how to efficiently train large-scale machine learning models for enormous output spaces, and obtain predictions in less than 1 millisecond for a data input with million labels, in the context of product recommendation and natural language processing.
We will also show the flexibility of dealing with diverse machine learning problems and data formats with assorted built-in utilities in PECOS.
By the end of the tutorial, we believe that attendees will be easily capable of adopting certain concepts to their own projects and address different machine learning problems with enormous output spaces.

* Presenters: Hsiang-Fu Yu (Amazon Search), Jiong Zhang (Amazon Search), Wei-Cheng Chang (Amazon Search), Jyun-Yu Jiang (Amazon Search), and Cho-Jui Hsieh (UCLA)

* Contributer: Wei Li (Amazon Search)

## Agenda

| Time | Session | Presenter | Material |
|---|---|---|---|
| 9:00 AM - 9:20 AM | Session 1: Introduction to PECOS | Dr. Hsiang-Fu Yu (Amazon) | [Slides](https://www.cs.utexas.edu/~rofuyu/talks/pecos-tutorial-kdd22-opening.pdf) |
| 9:20 AM - 10:00 AM | Environment Setup | | |
| 9:30 AM - 10:00 AM | Coffee Break | | |
| 10:00 AM - 10:30 AM  | Session 2: How to Apply PECOS to Extreme Multi-label Classification | Dr. Jyun-Yu Jiang (Amazon) | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%202%20Extreme%20Multi-label%20Classification%20with%20PECOS.ipynb) |
| 10:30 AM - 11:00 AM | Session 3: How to Perform Approximate Nearest Neighbor Search in PECOS | Dr. Wei-Cheng Chang (Amazon) | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%203%20Approximate%20Nearest%20Neighbor%20Search%20in%20PECOS.ipynb) |
| 11:00 AM - 11:20 AM | Session 4: Useful Utilities in PECOS | Dr. Jyun-Yu Jiang (Amazon) | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%204%20Utilities%20in%20PECOS.ipynb) |
| 11:20 AM - 11:40 AM | Session 5: How to Leverage Transformers in PECOS | Dr. Jiong Zhang (Amazon) | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%205%20eXtreme%20Multi-label%20Classification%20with%20XR-Transformer.ipynb) |
| 11:40 AM - 12:00 PM | Session 6: Research with PECOS | Prof. Cho-Jui Hsieh (UCLA & Amazon) | [Slides](http://web.cs.ucla.edu/~chohsieh/pecos_research.pdf) |


## Tutorial Instructions

###  Miniconda Installation
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Tutorial Material Execution
```bash
conda create -n tutorial_env python=3.9 -y
conda activate tutorial_env
python -m pip install libpecos==0.4.0 matplotlib panda requests jupyterlab
mkdir -p ~/pecos_tutorial_playground
cd ~/pecos_tutorial_playground
git clone https://github.com/amzn/pecos
python -m jupyterlab.labapp --ip=0.0.0.0 --port 8888 --no-browser --allow-root --notebook-dir=pecos/tutorials/kdd22
```
