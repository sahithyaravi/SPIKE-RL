<div align="center">

# 🧠 SPIKE-RL: Video-LLMs meet Bayesian Surprise

[![Paper](https://img.shields.io/badge/ICLR_2026-Paper-red)](https://iclr.cc/virtual/2026/poster/10009599)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Sahithya Ravi, Aditya Chinchure, Raymond T. Ng, Leonid Sigal, Vered Shwartz*

*The Fourteenth International Conference on Learning Representations (ICLR 2026)*

</div>

---

## 📖 Overview

Real-world videos often show routine activities punctuated by memorable, surprising events. However, most Video-LLMs process videos by sampling frames uniformly, likely missing critical moments that define a video's narrative.

We introduce **SPIKE**, an inference-time framework that quantifies **Bayesian Surprise** as the belief update triggered by new visual evidence in the video stream, identifying moments where new visual evidence conflicts with prior beliefs.

- 🔍 **SPIKE** effectively localizes surprise in videos, correlated with humans on positive ([FunQA](https://funqa-benchmark.github.io/)) and negative ([Oops!](https://oops.cs.columbia.edu/)) surprise benchmarks.
- 🤖 **SPIKE-RL** further improves on SPIKE's ability to detect surprise, leveraging GRPO to refine its belief hypotheses based on a reward signal from the video caption.
- 🎯 Both methods guide **query-agnostic surprise-weighted frame sampling**, allocating more frames to interesting moments — achieving consistent performance gains on five downstream benchmarks.

---

<!-- ## 🎬 Demo

<video src="assets/demo.mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

--- -->

## 🚀 Installation

### 1. Set up & Activate a Virtual Environment
```bash
virtualenv spike --python=python3.10.13
source spike/bin/activate
```

### 2. Install Core Dependencies
```bash
cd ..
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
```

### 3. Install `qwen-vl-utils`
```bash
cd qwen-vl-utils
pip install -e .
```

### 4. Install Custom Transformers & Remaining Dependencies

Download the custom transformers from [here](https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view) and run:

```bash
unzip transformers-main.zip
cd ./transformers-main
pip install .
```

```bash
pip uninstall transformers -y
pip install trl==0.16.0
pip install torchvision==0.21.0
pip install numpy==1.22.4
pip install sentence_transformers decord peft opencv_python wandb
```

---

## 📦 Data & Checkpoints

> The dataset and model checkpoints will be made available on [HuggingFace](https://huggingface.co) soon. Stay tuned!

---

## 📄 Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{
ravi2026spikerl,
title={{SPIKE}-{RL}: Video-{LLM}s meet Bayesian Surprise},
author={Sahithya Ravi and Aditya Chinchure and Raymond T. Ng and Leonid Sigal and Vered Shwartz},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=QLiXtWEAkq}
}
```

</div>