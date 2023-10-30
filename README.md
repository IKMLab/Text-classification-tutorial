# Text-Classification-tutorial

## Envrionment
- Ubuntu 20.04
- Python 3.9.5
- CUDA 10.1

## Get started
```bash
git clone https://github.com/IKMLab/Text-classification-tutorial
cd Text-classification-tutorial
pip install -r requirements.txt
```

## Download Dataset
- Currently support AG News dataset
    - Download link: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
```
mkdir data
mv archive.zip data/
cd data
unzip archive.zip
rm archive.zip
```

## Run EDA (Exploratory Data Analysis)
```
python simple_eda.py
```

## Run training / evaluations
```
python main.py \
--data_name agnews \
--use_agnews_title \
--model_name bert-base-uncased \
--max_length 128 \
--batch_size 32 \
--test_batch_size 128 \
--num_epoch 1 \
--learning_rate 3e-5
```
- This will also be provided in the `scripts/run_agnews.sh` file.