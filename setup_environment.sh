git clone https://huggingface.co/datasets/Az39/Code-Completion-Evaluation-Dataset-Package temp_repo
mkdir -p source/data
mv temp_repo/* temp_repo/.* source/data/
rm -rf temp_repo

python3.11 -m venv env
source env/bin/activate

pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
pip3 install nltk
python -m nltk.downloader punkt punkt_tab
pip3 install flash-attn --no-build-isolation
pip3 install typing_extensions==4.14.1
pip3 install pytest pytest-cov pytest-benchmark coincidence cssutils docutils openpyxl hypothesis pycodestyle