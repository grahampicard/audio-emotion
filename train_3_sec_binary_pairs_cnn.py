import argparse
import numpy as np
import os
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import f1_score

# pytorch imports
from torch import manual_seed
from torch import nn
from torch import optim
from torch.autograd import Variable # add gradients to tensors
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

# user-created files
from models.cnn_3_seconds_binary_classifier import CNN_simple_3s_32k
from source.data_loaders import load_section_level_stft


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def train(model, optimizer, dataloader, device, epoch, args):

    # set model to train mode
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        # https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()/len(data)))

    print("Epoch {} complete! Average training loss: {}".format(epoch,
        train_loss/len(dataloader.dataset)))

    return model


def valid(model, dataloader, device, args):

    model.eval()
    valid_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(output, target)
        valid_loss += loss.item()

    print("Average validation loss: {}".format(valid_loss/len(dataloader.dataset)))
    return valid_loss


def test(model, dataloader, device, args):

    model.eval()
    test_loss = 0.0

    n_correct, n_total = 0, 0
    y_preds, y_true = [], []

    for _, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        test_loss += loss.item()

        y_test_pred = torch.sigmoid(output)
        y_pred = torch.round(y_test_pred)

        y_preds.extend(y_pred.data.cpu().tolist())
        y_true.extend(target.data.cpu().tolist())
        n_correct += (y_pred == target).sum().detach().tolist()
        n_total += len(target)
        print(n_correct / n_total)

    print("Average test loss: {}".format(test_loss/len(dataloader.dataset)))
    print("Test accuracy: {}".format(100. * n_correct/n_total))
    print("Test correct #: {}".format(n_correct))
    print("Test F1 score: {}".format(100. * f1_score(np.asarray(y_true), np.asarray(y_preds),
                                                       average='weighted')))

    return y_preds, y_true

    #test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y>=0, pred>=0, average='binary')

def save_confusion(pred, true, file_name, annotations=True):

    df = pd.DataFrame(confusion_matrix(pred, true))

    if annotations:
        df.columns = 'true_0', 'true_1'
        df.index = 'pred_0', 'pred_1'
        df.index.name = 'pred'

    df.to_csv(file_name, index=True)


if __name__ == "__main__":

    # Include Hyperparameters for developing our Neural Network
    parser = argparse.ArgumentParser(description='Spectrogram + Emotion CNN')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', help='input batch size for testing (default: 50)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    #parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    #parser.add_argument('--momentum', type=float, default=0.1, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='simple')
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() & ~args.no_cuda else {}
    device = torch.device("cuda" if torch.cuda.is_available() & ~args.no_cuda else "cpu")

    if args.seed is not None:
        manual_seed(args.seed)

    labels = [x.replace('.csv', '') for x in os.listdir('./data/interim/expanded-3secondsegments/labels') if 'happy-not' in x]

    for label in labels:

        # Load training data
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_section_level_stft(label_type=label, valid_split=0.6, test_split=0.8)
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        valid_dataset = TensorDataset(valid_features, valid_labels)
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_dataset = TensorDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, **kwargs)

        # Instantiate model
        model = CNN_simple_3s_32k().to(device)
        optimizer = optim.Adam(model.parameters())

        min_valid_loss = float('Inf')

        for epoch in range(1, args.epochs + 1):
            curr_model = train(model, optimizer, train_loader, device, epoch, args)
            curr_valid_loss = valid(curr_model, valid_loader, device, args)
            if (curr_valid_loss < min_valid_loss):
                min_valid_loss = curr_valid_loss
                if (args.save_model):
                    #torch.save(curr_model.state_dict(),"./data/processed/cnn-boilerplate.pt")
                    torch.save(curr_model, f"./data/processed/cnn-binary-{label}-{args.model}-3s_32k.pt")
                print("Found new best model, saving to disk!")

        best_model = torch.load(f"./data/processed/cnn-binary-{label}-{args.model}-3s_32k.pt")
        pred, true = test(best_model, test_loader, device, args)
        save_confusion(pred, true, f"./data/processed/cnn-binary-{label}-{args.model}-3s_32k-confusion.csv")