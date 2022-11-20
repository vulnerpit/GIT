import torch
from torch.utils import data
from torchvision import transforms
import glob
from PIL import Image
import numpy as np

# Super parameter
batch_size = 1
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# Create dataset
all_imgs_path = glob.glob(r'E:\PyCharmProject\MNIST\data\train+test\*\*.png')  # 绝对路径

# label
species = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
all_labels = []
for img in all_imgs_path:
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)

# transform
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])


class MyDataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)


# 划分训练集和测试集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

s = int(len(all_imgs_path) * 0.8)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

train_ds = MyDataset(train_imgs, train_labels, transform)
test_ds = MyDataset(test_imgs, test_labels, transform)
train_dl = data.DataLoader(train_ds, batch_size, shuffle=True)
test_dl = data.DataLoader(train_ds, batch_size, shuffle=False)


# Design model using class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入
        x = self.fc(x)
        return x


model = Net()

# Construct loss and optimizer
function = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Train and Test CLASS
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_dl, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = function(outputs, target.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用为反向过程算梯度 省去这步
        for data in test_dl:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch + 1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)
