<p align="center">
  <img src="./pic/DALL-E_logo.png" alt="TREET - made with DALL·E" width="20%" height=auto >
</p>

# TREET: TRansfer Entropy Estimation via Transformers

[![python](https://img.shields.io/badge/Python-3.8.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This work proposes a novel transformer-based approach for estimating the transfer entropy (TE) for stationary processes, incorporating the state-of-the-art capabilities of transformers. 

TREET, transfer entropy estimation via transformers, is a neural network attention-based estimator of TE over continuous spaces. The proposed approach introduces the TE as a Donsker-Vardhan representation and shows its estimation and approximation stages, to prove the estimator consistency. A detailed implementation is given with elaborated modifications of the attention mechanism, and an estimation algorithm. In addition, estimation and optimization algorithm is presented with Neural Density Generator (NDG) as auxiliary model.

We demonstrated the algorithms in various sequence-related tasks, such as estimating channel coding capacity, while proving the relation between it to the TE, emphasizing the memory capabilities of the estimator, and presenting feature analysis on the Apnea disease dataset.

[TREET: TRanasfer Entropy Estimation via Transformers Paper]()

All results are presented in our paper.

---

## Algorithms and Implementations

**1. Estimation**

We present a novel algorithm to estimate TE for a given memory parameter `l`,

<p align="center">
<img src=".\pic\treet_scheme.png" width=80% height=auto alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall scheme for estimation process (TE first part).
</p>

<p align="center">
<img src=".\pic\treet_arch.png" width=70% height=auto alt="" align=center />
<br><br>
<b>Figure 2.</b> Overall architecture for estimation (TE first part).
</p>



**2. Optimization**

In addition, we use Neural Distribution Generator (NDG) for optimizing TE, while estimating it, with alternate learning procedure, 

<p align="center">
<img src=".\pic\treet_ndg_scheme.png" width=80% height=auto alt="" align=center />
<br><br>
<b>Figure 3.</b> Overall scheme for optimization and estimation process (TE first part).
</p>

<p align="center">
<img src=".\pic\ndg_arch.png" width=50% height=auto alt="" align=center />
<br><br>
<b>Figure 4.</b> NDG architecture for estimation (without feedback).
</p>

---

## Get Started

1. Install Python 3.8.10, PyTorch 2.0.1 and all requirments from `requierments.txt` file.
2. (Optional) Download data via script:
```bash
./datasets/apnea/downloader.sh
```
   - Check [here](https://physionet.org/content/santa-fe/1.0.0/) for further details about the Apnea dataset.
4. Train the model. We provide the experiment scripts of all benchmarks under the folder `./config`. You can reproduce the experiment results by:

```python
python run.py --config <file>.json5
```

## Citation

If you find this repo useful, please cite our paper. 

```
@misc{luxembourg2024treet,
      title={TREET: TRansfer Entropy Estimation via Transformer}, 
      author={Omer Luxembourg and Dor Tsur and Haim Permuter},
      year={2024},
      eprint={2402.06919},
      archivePrefix={arXiv},
      primaryClass={cs.IT}
}
```

## Contact

If you have any question or want to use the code, please contact [omerlux@post.bgu.ac.il](omerlux@post.bgu.ac.il).

## Acknowledgement

We appreciate the following sources for code base and datasets:

- Repository - https://github.com/thuml/Autoformer

- Apnea Data - https://physionet.org/content/santa-fe/1.0.0/ 

- Logo making - https://openai.com/dall-e/


