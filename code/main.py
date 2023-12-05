import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 创建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x): 
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
# 开始训练模型
def train():
    # 实例化一个网络
    network = Net().to(device)
    

    # 一个简单的优化器
    optimizer = optim.SGD(network.parameters(), lr=0.0125, momentum=0.5)

    epoch = 1    
    # 训练网络
    while(True):
        # 一个简单的进度条
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100,leave=False)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 梯度置零
            optimizer.zero_grad()
            
            output = network(data)

            # 计算损失参数并进行调整调整
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # 简单的进度条
            loop.set_description(f"Epoch {epoch}")
            loop.update(1)

        if epoch % 5 == 0:
            torch.save(network.state_dict(), 'model_{}.pth'.format(epoch))

            test(network,test_loader)

            if os.path.exists('model_{}.pth'.format(epoch-15)):
                os.remove('model_{}.pth'.format(epoch-15))
            print('\n第{}次训练已保存'.format(epoch))
        epoch+=1

# 测试模型的精确度
def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在测试阶段，不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\n测试模型的准确度: %f \n' % (correct / total))

# 自定义图片测试
def demo(img):
    # 加载模型

    # 加载并预处理图像
    text_list = []
    for i in range(20):
        image = Image.open(img).convert('L')  # 将图像转换为灰度图像
        #image = Image.open('test.jpg'.format(i)).convert('L')  # 将图像转换为灰度图像
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 调整图像大小
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化图像
        ])
        image = transform(image).unsqueeze(0).to(device)  # 添加一个批次维度

        # 运行模型
        output = model(image)
        prediction = output.argmax(dim=1)  # 获取概率最大的类别
        # print('第{}次识别的结果是'.format(i), prediction.item())

        
        text_list.append(prediction.item())
    #多次预测数值并二次求出识别概率最大的数字
    result = max(set(text_list), key=text_list.count)
    print("当前图片{0}中的数字为: => {1} ".format(img,result))



    return result



model = Net().to(device)
model.load_state_dict(torch.load('model_7730.pth'))


if __name__ == '__main__':

    # 加载MNIST数据集
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                './data/', train=True, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])),batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                './data/', train=False, download=True,transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])),batch_size=64, shuffle=True)
    
    
    # model = Net().to(device)
    # model.load_state_dict(torch.load('model_7730.pth'))
    
    # 自定义单张测试
    # img = input('图片路径')
    # demo(img)


    # 识别测试当前目录下的9张自定义照片
    for i in range(10):
        demo('自定义测试图像/ceshi/test{}.jpg'.format(i))
    

    # 此项进行训练模型
    # train()

    # 此项可以进行模型准确度估测
    # test(model,test_loader)

