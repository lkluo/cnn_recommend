# coding=utf-8

"""generate data for model training and inference"""

import os
import re
import json
import pandas as pd
from collections import OrderedDict
import random
import itertools
from nltk.tokenize import word_tokenize
# from nltk import sent_tokenize
from pprint import pprint
from segmenter import split_single
from data import write_to_tsv

random.seed(999)

text_filepath = "data/amazon_review_part"
xlsx2json_filepath = "data/amazon_review.json"
export_dir = "data/json_data"
spec_filepath = "data/specifications.xlsx"
to_spec_filepath = "data/spec_ref.json"

requirement_filepath = "data/all_data.json"

if not os.path.exists(export_dir):
    os.mkdir(export_dir)


def _convert_nan(text):
    return text if pd.notna(text) else ""


def _to_indexed_list(content_list, index):
    return [_convert_nan(l[index]) for l in content_list]


def save_to_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def fetch_xlsx(filepath):
    return pd.read_excel(filepath, sheet_name=None)


def get_raw_data(filepath):

    dfs = fetch_xlsx(filepath)

    output_dict = OrderedDict()
    for pc_serie, review in dfs.items():
        content_list = pd.Series.to_list(review)
        content_list = [l for l in content_list if l[0] != " "]

        output_dict[pc_serie] = {
            "review": _to_indexed_list(content_list, 0),
            "pos_keyword": _to_indexed_list(content_list, 2),
            "neg_keyword": _to_indexed_list(content_list, 3),
            "need": _to_indexed_list(content_list, 4),
        }
    return output_dict


def dump_to_json():
    output_dict1 = fetch_xlsx(text_filepath+"1.xlsx")
    output_dict2 = fetch_xlsx(text_filepath+"2.xlsx")

    output_dict1.update(output_dict2)
    save_to_json(output_dict1, xlsx2json_filepath)


def extract_requirement_data():

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    review_dict = load_json(xlsx2json_filepath)
    num_of_series = 0
    num_of_sentences = 0
    data_dict = OrderedDict()

    for pc_series, content in review_dict.items():

        write_to_text = [line for line in content["need"] if line]

        if not write_to_text:
            continue

        num_of_series += 1
        # further do sentence tokenization
        write_to_text = [l for line in write_to_text for l in split_single(line) if l]
        data_dict[pc_series] = write_to_text
        num_of_sentences += len(write_to_text)

        # with open(os.path.join(export_dir, "%s.txt" % pc_series), "w") as fw:
        #     num_of_sentences += len(write_to_text)
        #     fw.write("\n".join(write_to_text) + "\n")
    save_to_json(data_dict, requirement_filepath)

    print("Completing..\n\tnumber of PC series: %s\n\tnumber of sentences: %s" % (num_of_series, num_of_sentences))


# extract_requirement_data()


def get_spec_json(filepath):
    dfs = fetch_xlsx(filepath)
    asins = list(dfs["spec"])
    other_specs = pd.Series.to_list(dfs["spec"])
    specs = [asins] + other_specs
    spec_dict = OrderedDict()

    num_specs = len(specs)
    num_labels = len(specs[1])


    specs_list = []
    all_specs = OrderedDict()

    for j in range(num_labels):
        if j < 1:
            continue

        for i in range(num_specs):
            item = specs[i][0]
            item = re.sub(r"g\ card", "gpu", item).lower()
            item = re.sub(r"harddisk", "hdd", item).lower()
            spec_dict[item] = specs[i][j]

        specs_list.append(spec_dict.copy())
        all_specs[spec_dict["asin"]] = spec_dict.copy()

    save_to_json(all_specs, to_spec_filepath)
    print("spec dict saved at", to_spec_filepath)


# get_spec_json(spec_filepath)


def labeling_gpu(keyword):
    """
    NVIDIA GeForce GTX 1xxxx: 0
    NVIDIA other: 1
    Intel HD Graphics [45]xxx: 2
    Intel (U)?HD Graphics 6xxx: 3
    (integrated)? Intel HD Graphics other: 4
    AMD Radeon R2-4: 5
    AMD Radeon R5-7: 6
    other: 7
    """
    if re.compile(r"NVIDIA\ GeForce\ GTX", re.I).findall(keyword):
        return 0
    elif re.compile(r"NVIDIA\ GeForce\ (?=[^GTX])", re.I).findall(keyword):
        return 1
    elif re.compile(r"Intel\ HD\ Graphics\ (?=[45])", re.I).findall(keyword):
        return 2
    elif re.compile(r"Intel\ U?HD\ Graphics\ (?=[6])", re.I).findall(keyword):
        return 3
    elif re.compile(r"(integrated\ )?Intel\ HD\ Graphics\b", re.I).findall(keyword):
        return 4
    elif re.compile(r"AMD\ Radeon\ R[234]", re.I).findall(keyword):
        return 5
    elif re.compile(r"AMD\ Radeon\ R[567]", re.I).findall(keyword):
        return 6
    else:
        return 7

# gpu_label = labeling_gpu("integrated intel hd graphics")
# print(gpu_label)


def labeling_hdd(keyword):
    """
    1. SSD <=16: 0
    2. SSD (16, 32]: 1
    3. SSD (32, 128]: 2
    4. SSD (128, 256]: 3
    5. SSD (256, 512]: 4
    6. HDD<=256G: 5
    7. HDD (256G, 512]: 5
    8. HDD=512-1T: 6
    9. HDD>1T: 7
    10. other: 4
    """
    def _size():
        gb = re.compile(r"\d+(?=\sGB)|\d+\.[0]+").findall(keyword)
        if gb:
            return float(gb[0])
        else:
            tb = re.compile(r"\d+(?=\sTB)").findall(keyword)
            if tb:
                return float(tb[0]) * 1000
            else:
                return None
    def _type():
        if re.compile(r"SSD|Solid\ State|solid\_state", re.I).findall(keyword):
            return "SSD"
        else:
            return "HDD"

    # print(_size())
    size = _size()
    if not size:
        return 4
    if _type() == "SSD":
        if size <= 16:
            return 0
        elif 16 < size <= 32:
            return 0
        elif 32 < size <= 128:
            return 0
        elif 128 < size <= 256:
            return 1
        elif size > 256:
            return 2
    else:
        if size <= 256:
            return 3
        elif 256 < size <= 512:
            return 4
        elif 512 < size <= 1024:
            return 5
        elif size > 1024:
            return 6

# hdd_type = labeling_hdd("Flash Memory Solid State")
# print(hdd_type)


def labeling_ram(keyword):
    """
       "2 GB SDRAM": 0,
     "4 GB SDRAM DDR3": 0,
      "6 GB DDR SDRAM":1,
     "8 GB SDRAM DDR3": 2,
     "8 GB SDRAM DDR4": 3,
      "12 GB DDR SDRAM":4,
       "16 GB DDR4" :5,
       "others":6,
    """
    def _size():
        try:
            s = float(re.compile(r"\d+(?=\sGB)").findall(keyword)[0])
        except:
            s = None
        return s

    def _ddr():
        if re.compile(r"8\ GB\ SDRAM\ DDR3", re.I).findall(keyword):
            return 3
        elif re.compile(r"8\ GB\ SDRAM\ DDR4", re.I).findall(keyword):
            return 4
        else:
            return -1

    if not _size():
        return 0

    if _size() == 2:
        return 0
    elif _size() == 4:
        return 0
    elif _size() == 6:
        return 1
    elif _size() == 8:
        if _ddr() == 4:
            return 3
        else:
            return 2
    elif _size() == 12:
        return 4
    elif _size() == 16:
        return 5
    else:
        return 6

# ram_label = labeling_ram("flash_memory_solid_state")
# print(ram_label)


def labeling_cpu(keyword):
    """
    series: Celeron, i3, i5, i7, AMD A, other
    Celeron, ADM A, other:
        < 2 GHz: 0
        >=2, <3: 1
        >=3: 2
    i3:
        < 2.4: 3
        >= 2.4: 4
    i5:
        <=2: 5:
        >2, <3: 6
        >=3 (or no clock info) : 7
    i7:
        <=2: 6
        >2, <3: 7
        >3 (or no clock info): 8
    other: 9
    """
    def _series():
        if re.compile("i3", re.I).findall(keyword):
            return "i3"
        elif re.compile("i5", re.I).findall(keyword):
            return "i5"
        elif re.compile("i7", re.I).findall(keyword):
            return "i7"
        else:
            return "other"

    def _clock():
        clock = re.compile(r"(\d\.?\d?)(?=\ GHz)").findall(keyword)
        if clock:
            return float(clock[0])
        else:
            return None

    series = _series()
    clock = _clock()

    if not clock:
        return 9
    if series == "other":
        if clock < 2:
            return 0
        elif 2 <= clock < 3:
            return 1
        elif clock >= 3:
            return 2
    elif series == "i3":
        if clock < 2.4:
            return 3
        elif clock >= 2.4:
            return 4
    elif series == "i5":
        if clock <= 2:
            return 5
        elif 2 < clock < 3:
            return 6
        elif clock >= 3:
            return 7
    elif series == "i7":
        if clock <= 2:
            return 6
        elif 2 < clock < 3:
            return 7
        elif clock >= 3:
            return 8
    else:
        return 9

# cpu_label = labeling_cpu("8032")
# pprint(cpu_label)


def labeling_screen(keyword):
    screen_to_label = {
    10.1: 0,
    11.6: 0,
    12.3: 9,
    12.5: 9,
    13.3: 1,
    13.5: 9,
    14.0: 2,
    15.6: 3,
    17.3: 4,
    19.5: 5
    }
    def _screen_size():
        return float(re.compile(r"(\d+\.?\d+?)").findall(keyword)[0])
    try:
        label = screen_to_label[_screen_size()]
    except:
        label = 9
    return label

# screen_label = labeling_screen("17.3 inches")
# print(screen_label)


def _split(data, ratio):
    random.shuffle(data)
    dev_index = int((1 - ratio) * len(data))
    return data[:dev_index], data[dev_index:]


def _tokenize(sentence):
    return " ".join(word_tokenize(sentence.lower()))


def re_sample(data):
    output = []
    for sent in data:
        sent_list = split_single(sent)
        if len(sent_list) > 1:
            sent_list = list(itertools.permutations(sent_list))
            sent = [" ".join(s) for s in sent_list]
        else:
            sent = [sent]
        output += sent
    return output


def get_train_dev_test(output_path, dev_ratio=0.1):

    types = ["screen", "cpu", "ram", "hdd", "gpu"]
    data_dict = load_json(requirement_filepath)
    spec_dict = load_json(to_spec_filepath)

    for _type in types:
        train_data = OrderedDict()
        dev_data = OrderedDict()
        test_data = OrderedDict()
        train_inputs, train_outputs = [], []
        dev_inputs, dev_outputs = [], []
        test_inputs, test_outputs = [], []

        for pc_series, sentences in data_dict.items():
            spec_keyword = spec_dict[pc_series][_type]
            sentences = [_tokenize(l) for l in sentences]
            if not isinstance(spec_keyword, str):
                spec_keyword = str(spec_keyword)
            # print(spec_keyword)
            _label = eval("labeling_%s" % _type)(spec_keyword)
            # if _type == "ram" and _label == 0:
            #     print(pc_series)
            # _data = dict([(sent, _label) for sent in sentences])
            train_sent, dev_sent = _split(sentences, dev_ratio)
            dev_sent, test_sent = _split(dev_sent, 0.5)
            # train_sent = resample(train_sent)

            # train_data.update(dict([(sent, _label) for sent in train_sent]))
            # dev_data.update(dict([(sent, _label) for sent in dev_sent]))
            # test_data.update(dict([(sent, _label) for sent in test_sent]))
            train_label = [_label] * len(train_sent)
            dev_label = [_label] * len(dev_sent)
            test_label = [_label] * len(test_sent)
            # update
            train_inputs.extend(train_sent)
            dev_inputs.extend(dev_sent)
            test_inputs.extend(test_sent)
            train_outputs.extend(train_label)
            dev_outputs.extend(dev_label)
            test_outputs.extend(test_label)

            # _type_data.update(_data)

        # save_to_json(train_data, os.path.join(export_dir, "%s_train.json" % _type))
        # save_to_json(dev_data, os.path.join(export_dir, "%s_dev.json" % _type))
        # save_to_json(test_data, os.path.join(export_dir, "%s_test.json" % _type))

        # save to tsv data
        the_filepath = os.path.join(output_path, _type)
        if not os.path.exists(the_filepath):
            os.makedirs(the_filepath)
        write_to_tsv(sentences=train_inputs, labels=train_outputs, filepath=os.path.join("%s/train.tsv" % the_filepath))
        write_to_tsv(sentences=dev_inputs, labels=dev_outputs, filepath=os.path.join("%s/dev.tsv" % the_filepath))
        write_to_tsv(sentences=test_inputs, labels=test_outputs, filepath=os.path.join("%s/test.tsv" % the_filepath))


# get_train_dev_test(output_path="data/tsv")