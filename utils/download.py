import os
import requests
import argparse
import tarfile
import tqdm

wmt_2018_train = "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz"
# download eval and test data
#wmt_2018_eval = "http://data.statmt.org/wmt18/translation-task/dev.tgz"
wmt_2018_test = "http://data.statmt.org/wmt18/translation-task/test.tgz"

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

def parse_eavl_test_data(text):
    data = []
    # read the text from the file
    with open(text, "r", encoding="utf-8") as f:
        # process the text line by line
        for line in tqdm.tqdm(f.readlines(), desc="Parsing the eval & test data"):
            # if line begins with <seg id="*"> and ends with </seg>, get the  content between the html tags
            line = line.strip()
            if line.startswith("<seg id=") and line.endswith("</seg>"):
                # remove the html tags `<seg id="[0-9]*">` and `</seg>`
                #line = line.replace(r"<seg id=\"[0-9]*\">", "").replace("</seg>", "")
                line = line.replace("</seg>", "").split(">")[1]
                data.append(line)
    return data

def download(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(os.path.join(folder, "wmt")):
        os.makedirs(os.path.join(folder, "wmt"))
    # download the files
    # download en file and unzip
    train_response = requests.get(wmt_2018_train)
    open(os.path.join(folder, "wmt.tgz"), "wb").write(train_response.content)
    extract(os.path.join(folder, "wmt.tgz"), os.path.join(folder, "wmt/train"))

    # download eval file and unzip
    eval_response = requests.get(wmt_2018_test)
    open(os.path.join(folder, "wmt.tgz"), "wb").write(eval_response.content)
    extract(os.path.join(folder, "wmt.tgz"), os.path.join(folder, "wmt"))
    eval_en_data = parse_eavl_test_data("data/wmt/test/newstest2018-enzh-src.en.sgm")
    eval_zh_data = parse_eavl_test_data("data/wmt/test/newstest2018-enzh-ref.zh.sgm")
    test_en_data = parse_eavl_test_data("data/wmt/test/newstest2018-zhen-ref.en.sgm")
    test_zh_data = parse_eavl_test_data("data/wmt/test/newstest2018-zhen-src.zh.sgm")

    for f in os.listdir("data/wmt/test"):
        os.remove(os.path.join("data/wmt/test", f))

    with open(os.path.join("data/wmt/test", "dev.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(eval_en_data))
    with open(os.path.join("data/wmt/test", "dev.zh"), "w", encoding="utf-8") as f:
        f.write("\n".join(eval_zh_data))
    with open(os.path.join("data/wmt/test", "test.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_en_data))
    with open(os.path.join("data/wmt/test", "test.zh"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_zh_data))

    # remove the gz files
    os.remove(os.path.join(folder, "wmt.tgz"))

if __name__ == "__main__":
    args = parse_args()
    print("downloading the data......")
    download(args.folder)
