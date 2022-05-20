# AutoTower

The codes for paper *AutoTower: Efficient Neural Architecture Search for Two-Tower Recommendation*

## Datasets

Taking ML-100k as an example.

- Download raw data of ml100k in `https://grouplens.org/datasets/movielens/100k`.
- Use `codes/datasets/data/ml100k/process_ml100k.ipynb` to preprocess the raw data.
- Use `python codes/datasets/movielens100k.py` to create dataset for further searching.

Make sure save locations of all datasets.

## Train supernet

- Set the search space in `codes/configs/config.py`.
- Use `sh scripts/train_supernet.sh` to train the supernet, you need to set the data path in the scripts and some configurations in `codes/train_supernet.py`.

## Search architecture

Use `sh scripts/search_arch.sh` to search the architectures, you need to set the data path in the scripts and some configurations in `codes/train_search.py`, especially the supernet tag.

## License

The codes and models in this repo are released under the GNU GPLv3 license.