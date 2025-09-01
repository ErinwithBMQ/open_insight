from src.feature_extract import FeatureExtract
import os

def feature_extract(repo_name, version):
    FeatureExtract_instance = FeatureExtract(repo_name, version)
    result = FeatureExtract_instance.get_repo_all_mes()
    return result

if __name__ == "__main__":
    feature_extract("https://github.com/numpy/numpy","v2.3.2")