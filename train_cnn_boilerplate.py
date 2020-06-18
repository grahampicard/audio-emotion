import argparse
import torch

# pytorch imports
from torch import manual_seed
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
  
# user-created files
from models.cnn_boilerplate import CNN
from source.data_loaders import load_stft_data


def train(model, optimizer, dataloader, device, epoch, args):
    
    # set model to train mode
    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    return model


if __name__ == "__main__":

    # Include Hyperparameters for developing our Neural Network
    parser = argparse.ArgumentParser(description='Spectrogram + Emotion CNN')
    parser.add_argument('--batch-size', type=int, default=5, metavar='N', help='input batch size for training (default: 5)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manual_seed(args.seed)
    
    # Load training data
    train_features, train_labels, test_features, test_labels = load_stft_data(split=0.95)
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Instantiate model
    model = CNN().to(device)    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, device, epoch, args)
