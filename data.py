# coding = utf-8
import numpy as np
import torch
import json
import os
import pandas as pd
import csv


def get_batch(batch, word_vec, embed_size=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    # embed = np.zeros((max_len, len(batch), embed_size))
    embed = np.zeros((len(batch), max_len, embed_size))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            try:
                embed[i, j, :] = word_vec[batch[i][j]]
            except:
                embed[i, j, :] = word_vec['<p>']

    return torch.as_tensor(embed).float()


def get_word_dict(sentences):
    word_dict = {}
    for s in sentences:
        for w in s.split():
            if w not in word_dict:
                word_dict[w] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_vector(word_dict, glove_path):
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with glove vectors'.format(len(word_vec), len(word_dict)))
    # with open("vocab.vec", "w") as f:
    #     words = dict([(word,1) for word in word_dict if word not in word_vec])
    #     json.dump(words, f, indent=2, ensure_ascii=False)
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_vector(word_dict, glove_path)
    print('Vocab size: {0}'.format(len(word_vec)))
    return word_vec


def write_to_tsv(sentences, labels, filepath):
    df = pd.DataFrame({"sentence": sentences, "label": labels}, columns=["sentence", "label"])
    df.to_csv(filepath, sep="\t")


class DataProcessor:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, set_type):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "%s.tsv" % set_type))
        )

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @staticmethod
    def _create_examples(lines):
        examples = {
            "text": [],
            "label": []
        }
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text = line[1]
            label = int(line[2])
            examples["text"].append(text)
            examples["label"].append(label)
        examples["label"] = np.array(examples["label"])
        return examples

def get_data_ready(filepath, filename):
    def load_json(file):
        with open(file) as f:
            return json.load(f)

    def put_into_file(the_dict):
        target = np.array(list(the_dict.values()))
        return {'sent': list(the_dict.keys()), 'label': target}

    train_sent = load_json(os.path.join(filepath, "%s_train.json" % filename))
    dev_sent = load_json(os.path.join(filepath, "%s_dev.json" % filename))
    test_sent = load_json(os.path.join(filepath, "%s_test.json" % filename))

    return put_into_file(train_sent), put_into_file(dev_sent), put_into_file(test_sent)


if __name__ == '__main__':
    filepath = "data/tsv"
    item = "cpu"
    processor = DataProcessor(os.path.join(filepath, item))
    # train_examples = processor.get_examples(set_type="test")
    # print(train_examples)
    # print(len(train_examples["text"]))
    # print(len(train_examples["label"]))