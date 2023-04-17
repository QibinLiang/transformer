import os
import requests
import argparse
import tarfile

wmt_2018 = "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz"

# argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Download the WMT 2018 data")
    parser.add_argument("--folder", type=str, default="data", help="Folder to save the data")
    args = parser.parse_args()
    return args

def extract(tar_path, target_path):
    tar = tarfile.open(tar_path, "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, target_path)
    tar.close()

def download(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(os.path.join(folder, "wmt")):
        os.makedirs(os.path.join(folder, "wmt"))
    # download the files
    # download en file and unzip
    en_response = requests.get(wmt_2018)
    open(os.path.join(folder, "wmt.tgz"), "wb").write(en_response.content)
    extract(os.path.join(folder, "wmt.tgz"), os.path.join(folder, "wmt"))
    # remove the gz files
    os.remove(os.path.join(folder, "wmt.tgz"))

if __name__ == "__main__":
    args = parse_args()
    print("downloading the data......")
    download(args.folder)
