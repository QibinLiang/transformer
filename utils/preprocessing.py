import os
import re
import jieba
import nltk
import string
import argparse
import tqdm
import json
from nltk.tokenize import word_tokenize as en_tokenize
import logging

# todo: refactor the code, make it more readable

# turn off jieba logging
jieba.setLogLevel(logging.INFO)

# Tokenize the text
def tokenize(text, lang="en", zh_char_level=False):
    if lang == "en":
        return en_tokenize(text)
    else:
        if zh_char_level:
            return split_char(text)
        return list(jieba.cut(text))

def split_char(str):
	english = 'abcdefghijklmnopqrstuvwxyz0123456789'
	output = []
	buffer = ''
	for s in str:
		if s in english or s in english.upper():
			buffer += s
		else: 
			if buffer:
				output.append(buffer)
			buffer = ''
			output.append(s)
	if buffer:
		output.append(buffer)
	return output

# Remove stopwords
def remove_stopwords(text, lang="en"):
    if lang == "en":
        return [word for word in text 
                if word not in nltk.corpus.stopwords.words("english")]
    else:
        return text
    
def remove_punctuation(text, lang="en"):
    if lang == "en":
        # remove english punctuation
        return [word for word in text if word not in string.punctuation]
    else:
        # remove chinese punctuation
        return [word for word in text if word not in "，。！？、（）《》【】：；‘’…—"]

# process the text
def preprocess(text, lang="en", freq_threshold=2 ,keep_punc=False, zh_char_level=False):
    data = []
    tokens_freq = {}
    # read the text from the file
    with open(text, "r", encoding="utf-8") as f:
        # process the text line by line
        for line in tqdm.tqdm(f.readlines(), desc="Preprocessing "+lang+" data"):
            # remove the \n
            line = line.strip().lower()
            # remove blank characters
            # todo: improve this part
            if lang == "zh":
                # remove all the content in the brackets
                line = re.sub(r"\(.*?\)", "", line)
                line = re.sub(r"\（.*?\）", "", line)
                # remove spaces
                line = line.replace(" ", "")
                line = line.replace("　", "")
                line = line.replace(" ", "")
                # unrecognized characters
                line = line.replace("�", "")
                
            # tokenize the text
            line = tokenize(line, lang, zh_char_level)
            # remove the punctuation
            if not keep_punc:
                line = remove_punctuation(line, lang)
            # remove the stopwords
            # line = remove_stopwords(line, lang)
            # add the processed text to the data
            data.append(line)          
            # add the tokens to the set
            tokens_freq.update({token: tokens_freq.get(token, 0) + 1 for token in line})
    filtered_tokens = filter(lambda x: x[1] > freq_threshold, tokens_freq.items())
    tokens2id = {token: i+2 for i, (token, freq) in enumerate(filtered_tokens)}
    # add blank, unk and sos/eos tokens
    tokens2id["<blank>"] = 0
    tokens2id["<unk>"] = 1
    tokens2id["<sos/eos>"] = len(tokens2id)
    return data, tokens2id

def save_data(data, folder, dtype="train/train", lang="en", tok_level="word"):
    with open(os.path.join(folder, dtype+"_"+lang+"_"+tok_level+".txt"), "w", encoding="utf-8") as f:
        for line in data:
            f.write(" ".join(line) + "\n")

def save_dict(data, folder, lang="en", tok_level="word"):
    with open(os.path.join(folder, lang+"_dict_"+tok_level+".txt"), "w", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(key + " " + str(value) + "\n")

def save_json(en_data, zh_data, folder, dtype="train", tok_level="word"):
    with open(os.path.join(folder, dtype+"_"+tok_level+".json"), "w", encoding="utf-8") as f:
        for en, zh in zip(en_data, zh_data):
            json.dump({"src": zh, "tgt": en}, f,  ensure_ascii=False)
            f.write("\n")

# argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the WMT 2018 data")
    parser.add_argument("--folder", type=str, 
                        default="data/wmt", 
                        help="Folder to save the data")
    parser.add_argument("--keep_punc", action="store_true",
                        help="Keep the punctuation")
    parser.add_argument("--token_level", type=str,
                        default="word")
    parser.add_argument("download_punkt", action="store_true",
                        help="Download the punkt package")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.download_punkt:
        nltk.download('punkt')
    tok_level = args.token_level
    # preprocess the data
    print("preprocessing the train data......")
    en_data, en_tokens2id = preprocess(
        os.path.join(args.folder, 
                    "train/training-parallel-nc-v13/news-commentary-v13.zh-en.en"), 
                    "en")
     
    zh_data, zh_tokens2id = preprocess(
        os.path.join(args.folder, 
                    "train/training-parallel-nc-v13/news-commentary-v13.zh-en.zh"), 
                    "zh", 
                    zh_char_level=(tok_level == "char"))
    # save the data
    print("saving the train data......")
    save_data(en_data, args.folder,"train/train", "en", tok_level)
    save_data(zh_data, args.folder, "train/train","zh", tok_level)
    #save the dict
    save_dict(en_tokens2id, args.folder, "en", tok_level)
    save_dict(zh_tokens2id, args.folder, "zh", tok_level)
    # save the json
    save_json(en_data, zh_data, args.folder, "train", tok_level)

    print("preprocessing the dev data......")
    en_dev_data, _ = preprocess(
        os.path.join(args.folder, "test/dev.en"), "en")
    zh_dev_data, _ = preprocess(
        os.path.join(args.folder, "test/dev.zh"), "zh", zh_char_level=(tok_level == "char"))
        # save the data
    print("saving the dev data......")
    save_data(en_dev_data, args.folder,"test/dev", "en", tok_level)
    save_data(zh_dev_data, args.folder,"test/dev", "zh", tok_level)
    # save the json
    save_json(en_dev_data, zh_dev_data, args.folder,"dev", tok_level)

    print("preprocessing the test data......")
    en_test_data, _ = preprocess(
        os.path.join(args.folder, "test/test.en"), "en", keep_punc=args.keep_punc)
    zh_test_data, _ = preprocess(
        os.path.join(args.folder, "test/test.zh"), "zh", zh_char_level=(tok_level == "char"), keep_punc=args.keep_punc)
        # save the data
    print("saving the test data......")
    save_data(en_test_data, args.folder,"test/test", "en", tok_level)
    save_data(zh_test_data, args.folder,"test/test", "zh", tok_level)
    # save the json
    save_json(en_test_data, zh_test_data, args.folder,"test", tok_level)
