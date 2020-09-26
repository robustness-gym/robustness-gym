# Robustness Gym

### Installation
```
conda create -n rgym python=3.8
pip install -r requirements.txt
python -m spacy download en_core_web_sm
git clone https://github.com/robustness-gym/robustness-gym.git
cd robustness-gym/
pip install -e .
```


#### Installing Spacy GPU
To install Spacy with GPU support, use the installation steps given below.
```
pip install cupy
pip install spacy[cuda]
```

#### Installing neuralcoref
The standard version of `neuralcoref` does not use GPU for prediction and a pull request that is pending adds this 
functionality (https://github.com/huggingface/neuralcoref/pull/149). 
Follow the steps below to use this.   
```
git clone https://github.com/dirkgr/neuralcoref.git
cd neuralcoref
git checkout GpuFix
pip install -r requirements.txt
pip install -e .
```