import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from model import build_model
from utils import save_model, save_plots
from datasets import train_loader, valid_loader
from tqdm.auto import tqdm

# 构造参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='number of epochs to train our network for')
args = vars(parser.parse_args())

# 学习率
lr = 0.001
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
model = build_model(pretrained=True, fine_tune=False).to(device)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in enumerate(trainloader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(image)

        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        loss.backward()

        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)

            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc



train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# 开始训练

for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-' * 50)

# 保存模型
save_model(epochs, model, optimizer, criterion)
# 画图
save_plots(train_acc, valid_acc, train_loss, valid_loss)
