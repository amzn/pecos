# KDD 2022 Hands-on Tutorial - PECOS: Prediction for Enormous and Correlated Output Spaces

In this tutorial, we will introduce several key functions and features of the PECOS library.
By way of real-world examples, the attendees will learn how to efficiently train large-scale machine learning models for enormous output spaces, and obtain predictions in less than 1 millisecond for a data input with million labels, in the context of product recommendation and natural language processing.
We will also show the flexibility of dealing with diverse machine learning problems and data formats with assorted built-in utilities in PECOS.
By the end of the tutorial, we believe that attendees will be easily capable of adopting certain concepts to their own projects and address different machine learning problems with enormous output spaces.

* Presenters: Hsiang-Fu Yu (Amazon Search), Jiong Zhang (Amazon Search), Wei-Cheng Chang (Amazon Search), Jyun-Yu Jiang (Amazon Search), and Cho-Jui Hsieh (UCLA)

* Contributer: Wei Li (Amazon Search)

## Agenda

| Time | Session | Material |
|---|---|---|
| 9:00 AM - 9:20 AM | Session 1: Introduction to PECOS | |
| 9:20 AM - 10:00 AM | Environment Setup | |
| 9:30 AM - 10:00 AM | Coffee Break | | 
| 10:00 AM - 10:30 AM  | Session 2: eXtreme Multi-label Classification (XMC) with PECOS | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%202%20Extreme%20Multi-label%20Classification%20with%20PECOS.ipynb) |
| 10:30 AM - 11:00 AM | Session 3: Approximate Nearest Neighbor (ANN) Search in PECOS | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%203%20Approximate%20Nearest%20Neighbor%20Search%20in%20PECOS.ipynb) |
| 11:00 AM - 11:20 AM | Session 4: Utilities in PECOS | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%204%20Utilities%20in%20PECOS.ipynb) |
| 11:20 AM - 11:50 AM | Session 5: eXtreme Multi-label Classification (XMC) with XR-Transformer | [Notebook](https://github.com/amzn/pecos/blob/mainline/tutorials/kdd22/Session%205%20eXtreme%20Multi-label%20Classification%20with%20XR-Transformer.ipynb) |
| 11:50 AM - 12:00 AM | Session 6: Research with PECOS | |
| 12:00 PM - 12:10 PM | Closing Remarks | |

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

### Tutorial Mateiral Execution
```bash
conda create -n tutorial_env python=3.9 -y
conda activate tutorial_env
python -m pip install libpecos==0.4.0 matplotlib panda requests jupyterlab
mkdir -p ~/pecos_tutorial_playground
cd ~/pecos_tutorial_playground
git clone https://github.com/amzn/pecos
python -m jupyterlab.labapp --ip=0.0.0.0 --port 8888 --no-browser --allow-root --notebook-dir=pecos/tutorials/kdd22
```