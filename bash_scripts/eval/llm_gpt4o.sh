#!/bin/bash
source ../.bashrc
source .env
source ../videobpo/bin/activate
python -u src/open_r1_video/llm_match_gpt4o.py --json_path results/captions/oops/64/blackswan_uniform.jsonl