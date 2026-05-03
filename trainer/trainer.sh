#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace
source "/root/workspace/env/bin/activate"
python trainer.py > "/root/workspace/versions_raio_changed/instinctguo__llama3.1-8b-train-token/trainer.log" 2>&1
deactivate
