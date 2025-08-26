import csv
from tqdm import tqdm
from . import FeatureExtract

if __name__ == "__main__":
    repo_url = "https://github.com/numpy/numpy"
    version = "v1.21.0"
    feature_extract = FeatureExtract(repo_url, version)
    res = feature_extract.get_repo_all_mes()