import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")
from resnet.resnet_class import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/media/sata/meepo/data/cifar-10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='/media/sata/meepo/data/cifar-10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# class Net(nn.Module):
#
#     def __init__(self):
#         # question: super function? __init__ function?
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

device = torch.device("cuda:0")
net = ResNet(ResidualBlock)
# if torch.cuda.device_count() > 1:
#     print("Let's use, ", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)

LR = 1e-2
net = nn.DataParallel(net)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

def adjust_learning_rate(optimizer, epoch):
    if epoch >= 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3


for epoch in range(135):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs, data is a list of [inputs, labels]
        # inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = data
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # images, labels = data[0].to(device), data[1].to(device)
            images, labels = data
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f" epoch: {epoch} Accuracy of the network on the 10000 test images: {100 * correct / total}%")
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    # adjust_learning_rate(optimizer, epoch)




print('Finished Training')
#
PATH = '/media/sata/meepo/data/cifar-10/model/cifar_net.pth'
torch.save(net.state_dict(), PATH)


# dataiter = iter(testloader)
# data = dataiter.next()
# images, labels = data[0].to(device), data[1].to(device)
#
# imshow(torchvision.utils.make_grid(images))
# print("GroundTruth: ", ''.join('%s ' % classes[labels[j]] for j in range(4)))
#
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ''.join('%s ' % classes[predicted[j]] for j in range(4)))


# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
#
# class_correct = list(0. for i in  range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
#
# for i in range(10):
#     print(f"Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]}")

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)
# net.to(device)

