Robustness Gym
================================
[![Build Status](https://travis-ci.com/robustness-gym/robustness-gym.svg?token=T6CDNeky2ippe6wEJvRV&branch=master)](https://travis-ci.com/robustness-gym/robustness-gym)
[![codecov](https://codecov.io/gh/robustness-gym/robustness-gym/branch/master/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/robustness-gym/robustness-gym)
[![license](https://img.shields.io/github/license/robustness-gym/robustness-gym)](https://github.com/robustness-gym/robustness-gym/LICENSE)

Robustness Gym is an evaluation toolkit for natural language processing.

### Installation
```
git clone https://github.com/robustness-gym/robustness-gym.git
cd robustness-gym/

conda create -n rgym python=3.8
conda activate rgym

pip install -r requirements.txt
pip install -e .
```


#### Installing Spacy GPU
To install Spacy with GPU support, use the installation steps given below.
```
pip install cupy
pip install spacy[cuda]
python -m spacy download en_core_web_sm
```

#### Installing neuralcoref
The standard version of `neuralcoref` does not use GPUs for prediction and a pull request that is pending adds this 
functionality (https://github.com/huggingface/neuralcoref/pull/149). 
Follow the steps below to use this.   
```
git clone https://github.com/dirkgr/neuralcoref.git
cd neuralcoref
git checkout GpuFix
pip install -r requirements.txt
pip install -e .
```

#### Progress bars in Jupyter 
Enable the following Jupyter extensions to display progress bars properly. 
```
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

#### TextBlob setup
Download and install the corpora that textblob uses.
```
python -m textblob.download_corpora
```
