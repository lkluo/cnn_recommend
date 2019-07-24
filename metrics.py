# coding=utf-8
import numpy as np
import os, json
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from data import DataProcessor
import argparse

sess = tf.InteractiveSession()

class Metrics:

    def __init__(self, output_dir, data_dir, item):
        data_dir = os.path.join(data_dir, item)
        output_dir = os.path.join(output_dir, item)

        try:
            predictions = np.load(os.path.join(output_dir, "predictions.npy"))
            y_pred = np.array(predictions).astype(np.float32)
            # y_pred = np.argsort(-y_pred, 1).astype(np.int64)
        except:
            y_pred = []

        try:
            data_processor = DataProcessor(data_dir)
            test_examples = data_processor.get_examples("test")
            labels = test_examples["label"]
            # labels = [[l] for l in labels]
            y_true = np.array(labels).astype(np.int64)
        except:
            y_true = []
            test_examples = {"text": [], "label": []}

        self.y_true = y_true
        self.y_preds = y_pred
        self.test_eamples = test_examples
        self.num_classes = len(set(y_true))

    def __call__(self):
        results = {}
        for k in range(1, self.num_classes+1):
            results[k] = self.get_top_k_scores(k)

        # saving classification report
        # class_report = classification_report(y_true=self.y_true.copy(), y_pred=self.y_preds.copy()[:, 0])
        return results, []

    def get_top_k_scores(self, k, y_true=None, y_preds=None):

        if y_true is None or y_preds is None:
            y_true = self.y_true.copy()
            y_preds = self.y_preds.copy()

        # convert to tensorflow identify
        y_preds = tf.identity(y_preds)
        y_true = tf.identity(y_true)

        _, precision = tf.metrics.precision_at_k(
            labels=y_true,
            predictions=y_preds,
            k=k)
        _, recall = tf.metrics.recall_at_k(
            labels=y_true,
            predictions=y_preds,
            k=k)

        result = {}
        sess.run(tf.local_variables_initializer())
        precision = np.around(sess.run(precision), 3)
        recall = np.around(sess.run(recall), 3)
        result["precision_micro"] = precision
        result["recall_micro"] = recall

        # result = {}
        # y_pred = y_preds[:, 0]
        # for i in range(len(y_true)):
        #     label = y_true[i]
        #     if label in y_preds[i][:k]:
        #         y_pred[i] = label
        #
        # # scores = classification_report(y_pred=y_pred, y_true=y_true, digits=3)
        # precision = precision_score(y_true, y_pred, average="macro")
        # recall = recall_score(y_true, y_pred, average="macro")
        # f1 = f1_score(y_true, y_pred, average="macro")
        # result["precision_macro"] = "{:.3f}".format(precision)
        # result["recall_macro"] = "{:.3f}".format(recall)
        # result["f1_macro"] = "{:.3f}".format(f1)
        #
        # precision = precision_score(y_true, y_pred, average="micro")
        # recall = recall_score(y_true, y_pred, average="micro")
        # f1 = f1_score(y_true, y_pred, average="micro")
        # result["precision_micro"] = "{:.3f}".format(precision)
        # result["recall_micro"] = "{:.3f}".format(recall)
        # result["f1_micro"] = "{:.3f}".format(f1)

        return result

    def get_long_short_text_scores(self):
        lengths = []
        texts, labels = self.test_eamples["text"], self.test_eamples["label"]
        for text in texts:
            l = len(text.split())
            lengths.append(l)

        m_l = np.median(lengths)
        result = {}
        for text_type in ["long", "short"]:
            if text_type == "long":
                index = [i for i, l in enumerate(lengths) if l >= m_l]
            else:
                index = [i for i, l in enumerate(lengths) if l < m_l]
                # y_true = [self.y_true[i] for i, l in enumerate(lengths) if l < m_l]
                # y_preds = [self.y_preds[i, :] for i, l in enumerate(lengths) if l < m_l]

            # print(y_preds.shape)
            y_true = self.y_true[index]
            y_preds = self.y_preds[index, :]
            scores = self.get_top_k_scores(k=1, y_true=y_true, y_preds=y_preds)
            # print("%s text, score:" % text_type, scores)
            result[text_type] = scores
        return result


# def get_top_k_scores(y_true, y_preds, k):
#     result = {}
#     y_pred = y_preds[:, 0]
#     for i in range(len(y_true)):
#         label = y_true[i]
#         if label in y_preds[i][:k]:
#             y_pred[i] = label
#
#     # scores = classification_report(y_pred=y_pred, y_true=y_true, digits=3)
#     precision = precision_score(y_true, y_pred, average="macro")
#     recall = recall_score(y_true, y_pred, average="macro")
#     result["precision_macro"] = "{:.3f}".format(precision)
#     result["recall_macro"] = "{:.3f}".format(recall)
#     precision = precision_score(y_true, y_pred, average="micro")
#     recall = recall_score(y_true, y_pred, average="micro")
#     result["precision_micro"] = "{:.3f}".format(precision)
#     result["recall_micro"] = "{:.3f}".format(recall)
#
#     return result


def data_summary():
    data_dir = os.path.join("data/tsv/cpu")
    data_processor = DataProcessor(data_dir)

    with open("output/summary.txt", "w") as fw:
        train_examples = data_processor.get_examples("train")
        dev_examples = data_processor.get_examples("dev")
        test_examples = data_processor.get_examples("test")
        texts = train_examples["text"] + dev_examples["text"] + test_examples["text"]

        length = [len(l.split()) for l in texts]
        max_len = np.max(length)
        min_len = np.min(length)
        median_len = np.median(length)
        num_words = sum(length)
        num_train = len(train_examples["text"])
        num_dev = len(dev_examples["text"])
        num_test = len(test_examples["text"])
        num_total = num_train + num_dev + num_test

        output = "total: %s\ntrain set: %s\ndev set:%s\ntest set:%s\n" \
                 "number of tokens:%s\nmax len:%s\nmin len:%s\nmedian len:%s\n" % (
            num_total, num_train, num_dev, num_test, num_words, max_len, min_len, median_len)
        print(output)
        fw.write(output)

        length = np.array(length)
        np.save("output/length.npy", length)


def main():
    parser = argparse.ArgumentParser("Metrics for text classification")
    parser.add_argument("--data_dir", type=str, default="data/tsv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--items", type=str, default="cpu,ram,gpu,hdd,screen")
    args = parser.parse_args()

    def save_to_json(data, filepath):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    items = args.items.split(",")
    # items = ["cpu"]
    scores = {}
    long_short_scores = {}
    class_reports = {}
    for item in items:
        metrics_ = Metrics(data_dir=args.data_dir, output_dir=args.output_dir, item=item)
        score, class_report = metrics_()
        scores[item] = score
        class_reports[item] = class_report
        long_short_score = metrics_.get_long_short_text_scores()
        long_short_scores[item] = long_short_score

    save_to_json(scores, os.path.join(args.output_dir, "scores.json"))
    save_to_json(class_reports, os.path.join(args.output_dir, "class_reports.json"))
    save_to_json(long_short_scores, os.path.join(args.output_dir, "scores_by_length.json"))


if __name__ == '__main__':
    main()
    # measure = Metrics(data_dir="data/tsv", output_dir="output", item="cpu")
    # result = measure(k=4)
    # print(result)
    # result = measure.get_long_short_text_scores()
    # print(result)
    # data_summary()