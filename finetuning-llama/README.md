# How to finetune an LLM on a personal dataset

## Below are steps to create a training dataset from a set of Youtube videos. In this example, algebra course videos are used to represent typical math lessons.

### Prerequisite
Install necessary packages:
pip install -r requirements.txt

1) Transcribe a set of youtube videos from youtube [playlist](https://www.youtube.com/watch?v=VXzm8ReImG0&list=PLgIi4lM74yW0ChmzTdT1w5ruCnqP0bv3J&index=2)
```
python transcribe.py
```

2) Setup a VLLM server running online Llama3.3-70B in another terminal window
```
VLLM_SKIP_WARMUP=true vllm serve meta-llama/Llama-3.3-70B-Instruct --task generate --trust-remote-code --tensor-parallel 4 --max_model_len 16384
```

3) Create a labelled dataset from your transcribed algebra lessons
```
python bootstrap_lessons.py
```

4) Launch unsloth QLoRA finetuning script on the resulting labelled data
```
jupyter lab unsloth_llama3_8B_SFT.ipynb
```
