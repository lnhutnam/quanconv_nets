import torch
import torch.nn as nn
import torch.optim as optim

from models import Net
from data_loader import train_loader, test_loader

batch_size = 256

use_cuda = torch.cuda.is_available()

if use_cuda:
    model = Net().cuda()
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    model = Net()
    loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
            output = model(data).cuda()
        else:
            output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()

        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))

model.eval()
with torch.no_grad():

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
            output = model(data).cuda()
        else:
            output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100 / batch_size)
    )
