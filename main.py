# coding=utf-8
import torch
import sys
import os
import pickle
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import recall_score, accuracy_score

from model import CNNText
from utils import CrossEntropyLoss
from data import *

import argparse


def main():
    parser = argparse.ArgumentParser("CNN for text classification")
    # data config
    parser.add_argument("--data_dir", type=str, default="data/tsv")
    parser.add_argument("--item", type=str, default="hdd")
    # environment config
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--glove_path", type=str, default=os.path.expanduser("~/data/glove/glove.840B.300d.txt"))
    # model config
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
    parser.add_argument("--filter_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    # train config
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.002, help="learning rate")
    parser.add_argument("--log_freq", type=int, default=100, help="")
    parser.add_argument("--val_freq", type=int, default=8, help="")
    parser.add_argument("--save_freq", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument('--early_stop', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1999)
    # action
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)

    # arguments
    args = parser.parse_args()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, args.item)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # for reproduce
    np.random.seed(args.seed)

    # get train and dev dataset
    data_dir = os.path.join(args.data_dir, args.item)
    data_processor = DataProcessor(data_dir)
    train_examples = data_processor.get_examples("train")
    full_sets = ["train_examples", "dev_examples", "test_examples"]
    try:
        dev_examples = data_processor.get_examples("dev")
    except Exception as e:
        dev_examples = None
        full_sets = ["train_examples", "test_examples"]
    test_examples = data_processor.get_examples("test")

    # get all texts for creating word2vec
    if dev_examples is None:
        texts = train_examples["text"] + test_examples["text"]
    else:
        texts = train_examples["text"] + dev_examples["text"] + test_examples["text"]
    num_classes = len(set(train_examples["label"]))

    print("Number of training examples:", len(train_examples["text"]))
    print("Number of classes:", num_classes)

    # word vector
    global WORD_VECTOR
    WORD_VECTOR = build_vocab(texts, args.glove_path)
    # convert to vectors
    for set_type in full_sets:
        eval(set_type)["text"] = np.array([[word for word in sent.split() if word in WORD_VECTOR] for sent in eval(set_type)["text"]]) #+ ['</s>']

    print("looking at item:", args.item)

    if args.train:
        print("model training..")
        # init model
        model = CNNText(
            embed_size=args.embed_size,
            num_class=num_classes,
            filter_size=args.filter_size,
            kernel_size=args.kernel_sizes,
            dropout=args.dropout
        )
        train(model=model, trainset=train_examples, devset=dev_examples, args=args)

    if args.predict:
        print("predicting..")
        model = CNNText(
            embed_size=args.embed_size,
            num_class=num_classes,
            filter_size=args.filter_size,
            kernel_size=args.kernel_sizes,
            dropout=0.0
        )
        if args.cuda:
            model.cuda()

        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_steps_0.pt")))
        output = predict(testset=test_examples, model=model, args=args)
        print(output.shape)


def train(trainset,
          model,
          args,
          devset):

    global WORD_VECTOR

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    loss_fn = CrossEntropyLoss(smooth_eps=args.smoothing)

    if args.cuda:
        model.cuda()
        loss_fn.cuda()

    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        # shuffle the data
        permutation = np.random.permutation(len(trainset["text"]))
        trainset["text"] = trainset["text"][permutation]
        trainset["label"] = trainset["label"][permutation]

        for steps in range(0, len(trainset["text"]), args.batch_size):
            feature = get_batch(trainset["text"][steps:steps + args.batch_size], WORD_VECTOR, embed_size=args.embed_size)
            target = trainset["label"][steps:steps + args.batch_size]

            # seq_len, batch_size, embed_size = feature.size()
            # feature = feature.view(-1, (batch_size, seq_len, embed_size))

            feature = Variable(feature.cuda()) if args.cuda else Variable(feature.cpu())
            target = Variable(torch.LongTensor(target)).cuda() if args.cuda else Variable(torch.LongTensor(target)).cpu()

            # if args.cuda:
            #     feature = Variable(feature.cuda())
            #     target = Variable(torch.LongTensor(target)).cuda()

            optimizer.zero_grad()
            logit = model(feature)

            loss = loss_fn(logit, target)
            loss.backward()
            optimizer.step()

            if steps % args.log_freq == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data,
                                                                             accuracy,
                                                                             corrects,
                                                                             args.batch_size))
            if devset and steps % args.val_freq == 0:
                dev_acc = evaluation(devset, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    last_epoch = epoch
                    # if args.save_best:
                    if True:
                        save(model, args.save_dir, 'best', "0")
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_freq == 0:
                save(model, args.save_dir, 'snapshot', steps)

    if devset:
        print("Best evaluate acc is %s reached at step %s epoch: %s" % (best_acc.tolist(), last_step, last_epoch))
    else:
        print("saving the last model..")
        save(model, args.save_dir, "best", "0")


def evaluation(devset, model, args):
    global WORD_VECTOR
    model.eval()
    corrects, avg_loss = 0, 0

    for steps in range(0, len(devset["text"]), args.batch_size):
        feature = get_batch(devset["text"][steps:steps + args.batch_size], WORD_VECTOR, embed_size=args.embed_size)
        target = devset["label"][steps:steps + args.batch_size]
        # seq_len, batch_size, embed_size = feature.size()
        # feature = feature.view(batch_size, seq_len, embed_size)

        feature = Variable(feature.cuda()) if args.cuda else Variable(feature.cpu())
        target = Variable(torch.LongTensor(target)).cuda() if args.cuda else Variable(torch.LongTensor(target)).cpu()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(devset["text"])
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    # print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def predict(testset, model, args):
    global WORD_VECTOR
    model.eval()

    feature = get_batch(testset["text"], WORD_VECTOR, embed_size=args.embed_size)
    target = testset["label"]
    feature = Variable(feature.cuda()) if args.cuda else Variable(feature.cpu())
    # target = Variable(torch.LongTensor(target)).cuda() if args.cuda else Variable(torch.LongTensor(target)).cpu()

    output = model(feature)
    # save
    # np.savetxt(os.path.join(args.save_dir, "predictions.txt"), output.data.numpy(), fmt="%f")
    np.save(os.path.join(args.save_dir, "predictions.npy"), output.data.numpy())
    _, predicted = torch.max(output, 1)

    _, index = torch.sort(output, 1)
    index = [l[::-1] for l in index.tolist()]

    target = target.tolist()

    num_class = len(index[0])
    recall = []
    for i in range(num_class):
        rank = i + 1
        new_predict = predicted.tolist()
        for j in range(len(target)):
            if target[j] in index[j][:rank]:
                new_predict[j] = target[j]

        _recall = recall_score(target, new_predict, average='weighted')
        recall.append(round(_recall, 2))
        print(recall)


    # print(target.tolist())
    # print(predicted.tolist())

    # acc = accuracy_score(target.tolist(), predicted.tolist())
    # recall = recall_score(target.tolist(), predicted.tolist(), average='weighted')

    return output


if __name__ == '__main__':
    main()