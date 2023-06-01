# PINA: Leveraging Side Information in eXtreme Multi-label Classification via Predicted Instance Neighborhood Aggregation, ICML 2023

This folder contains code to train XR-Transformer+PINA models and reproduce experiments 
in ["PINA: Leveraging Side Information in eXtreme Multi-label Classification via Predicted Instance Neighborhood Aggregation"](https://arxiv.org/pdf/2305.12349.pdf).

## Getting Started
* Clone the repository and enter `examples/pina` directory. 
* First create a [virtual environment](https://docs.python.org/3/library/venv.html) and then install dependencies 
by running the following command:
```bash 
pip install libpecos pandas gdown urllib3==1.26.6 
``` 
If you're unfamiliar with Python virtual environments, check out the 
[user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

* Install [pyxclib](https://github.com/kunaldahiya/pyxclib)
* Verify pytorch and CUDA:
```bash
 python -c "import torch; print('torch={}, cuda={}'.format(torch.__version__, torch.cuda.is_available()))"
```

## Training and Evaluation
To train and evaluate PINA+XR-Transformer model, run
``` bash
chmod a+x ./scripts/*
DATASET="LF-Amazon-131K"
bash scripts/run_pina.sh ${DATASET}
``` 
Recommended platform for training: [AWS p3.16xlarge instance](https://aws.amazon.com/ec2/instance-types/p3/) or equivalent.

### References:

[1] XC Repo: http://manikvarma.org/downloads/XC/XMLRepository.html
