# 0.1
Interpreting Large Pre-trained Language Models From The Training Set

### Major Changes
* Runnable docker image and Kubernetes config
* Added GPT2 model
* Added `notebooks/`
* Added wikitext-2 article sanity check experiment
* Added `utils/exact.py` for computing exact influences.

### Minor Changes
* Dockerfile
    - Added `wget`, `unzip`, `psmisc`, `vim`
    - Added `vimrc`

* Requirements
    - Added `transformers` (`v2.11.0`)


# 0.2
Switching gear towards downstream tasks

### Major Changes
* Have an initial running version
* Added `influence_utils.faiss_utils`
* Adding Faiss [dependency](https://github.com/kyamagu/faiss-wheels)
* Added support for `textattacks`
* Fixed a bug where `bert.pooler` parameters were filted during the gradient computation. The bug was caused by that `bert.pooler` parameters are not used in pre-trained LMs, and thus they were filted in the early stages of the codebase where the experiments are still performed on LMs. In the fine-tuned BERT models, however, `bert.pooler` parameters are used.
* Added nvidia-drivers to Dockerfile, and bumped `torch` base docker image from `1.5.0` to `1.5.1`


# 0.3

### Major Changes
* Splitting the implementation of sequential vs parallel influence calculation.
* Added customized Datasets
