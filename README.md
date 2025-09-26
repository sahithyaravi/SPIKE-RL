# SPIKE-RL: Video-LLMs meet Bayesian Surprise
*Sahithya Ravi, Aditya Chinchure, Raymond T. Ng, Leonid Sigal, Vered Shwartz*

Real-world videos often show routine activities punctuated by memorable, surprising events. However, most Video-LLMs process videos by sampling frames uniformly, likely missing critical moments that define a video's narrative. We introduce SPIKE, an inference-time framework that quantifies Bayesian Surprise as the belief update triggered by new visual evidence in the video stream, identifying moments where new visual evidence conflicts with prior beliefs. SPIKE effectively localizes surprise in videos, correlated with humans on positive (FunQA) and negative (Oops!) surprise benchmarks. SPIKE-RL further improves on SPIKE's ability to detect surprise, leveraging GRPO to refine its belief hypotheses based on a reward signal from the video caption. SPIKE and SPIKE-RL guide query-agnostic surprise-weighted frame sampling, which allocates more frames to interesting moments in the video. With this strategy, we achieve consistent performance gains on five downstream benchmarks. By enabling Video-LLMs to track beliefs and register surprise, our work paves the way for more robust models that can revise their understanding in response to new information.

## Instructions

## 1. Set up a Virtual Environment
```bash
virtualenv videobpo --python=python3.10.13
```


## 2. Activate the Virtual Environment
```bash
source videobpo/bin/activate
```

## 3. Install Core Dependencies
```bash
cd ..
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
```

## 4. Install `qwen-vl-utils`
```bash
cd qwen-vl-utils
pip install -e .
```

## 8. Install right transformers
```bash
pip uninstall transformers
```

Download "https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view".
```bash
unzip transformers-main.zip
cd ./transformers-main
pip install .
pip install trl==0.16.0
pip install torchvision==0.21.0
pip install numpy==1.22.4
pip install sentence_transformers
pip install decord
pip install peft
pip install opencv_python
pip install wandb
```
