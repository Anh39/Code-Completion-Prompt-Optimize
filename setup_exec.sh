wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

git lfs install
git clone https://huggingface.co/datasets/CSJianYang/ExecRepoBench

git clone --filter=blob:none --sparse https://github.com/QwenLM/Qwen2.5-Coder.git
cd Qwen2.5-Coder
git sparse-checkout set qwencoder-eval/base/benchmarks/ExecRepoBench
cd ..

mv Qwen2.5-Coder/qwencoder-eval/base/benchmarks/ExecRepoBench/* ExecRepoBench/

unzip ExecRepoBench/repos.zip -d ExecRepoBench

export PATH="$HOME/miniconda3/bin:$PATH"
source env/bin/activate
cd ExecRepoBench
cp ../create_test_repo.py create_test_repo.py
pip3 install typing_extensions==4.14.1
python create_test_repo.py --action "prepare_environments" --conda_dir "$HOME/miniconda3"
pip3 install typing_extensions==4.14.1
cd ..
deactivate

source /root/miniconda3/etc/profile.d/conda.sh
for env in /root/workspace/ExecRepoBench/envs/envs/*; do
    echo "Processing $env"
    conda activate "$env"
    pip3 install -q pytest pytest-cov pytest-benchmark coincidence cssutils docutils openpyxl hypothesis pycodestyle
    conda deactivate
done
deactivate

source env/bin/activate
python create_test_script.py
cd ExecRepoBench
python create_test_repo.py \
    --eval_workers 16 \
    --action "verify_all_repo_correctness" \
    --conda_dir "$HOME/miniconda3" \
    --env_dir /root/workspace/ExecRepoBench/envs/envs \
    --repo_root_path /root/workspace/ExecRepoBench/repos
cd ..
