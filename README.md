To clone the repo:
~~~bash
git clone https://github.com/datascience-labs/cdrl4ad.git && cd cdrl4ad
~~~

Install dependencies (virtual env is recommended):
~~~bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
~~~

To train:
~~~
python train.py --dataset <dataset> --epochs <epoch>
~~~
