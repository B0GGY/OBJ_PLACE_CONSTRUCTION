import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class PlaceNet(nn.Module):
    def __init__(self):
        super(PlaceNet, self).__init__()
        # 加载 ResNet18 模型，使用预训练权重
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Identity()
        self.new_layers = nn.Sequential(
            nn.BatchNorm1d(512*2),
            nn.ReLU(),
            nn.Linear(512*2, 512),  # 输入特征是 ResNet18 的输出 (512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # 输入特征是 ResNet18 的输出 (512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # 输入特征是 ResNet18 的输出 (512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # 输入特征是 ResNet18 的输出 (512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2),  # 假设最终分类 10 类
            nn.Softmax(dim=1) # 好像多了一个softmax层！！！！！
        )
        # resnet18 = models.resnet18(pretrained=False)
        #
        # self.resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
        # print(self.resnet18)

    def forward(self, x):
        x_channel = x.shape[1]//2
        x1 = x[:,:x_channel,:,:]
        x2 = x[:,x_channel:,:,:]
        emb1 = self.resnet18(x1)
        emb2 = self.resnet18(x2)
        emb_concat = torch.concat((emb1, emb2),dim=1)
        # output = F.softmax(self.new_layers(emb_concat), dim=1)
        output = self.new_layers(emb_concat)

        # print(emb1.shape)
        # # print(emb2.shape)
        # print(emb_concat.shape)
        # print(test_output.shape)
        return output


if __name__ == '__main__':
    test_net = PlaceNet()
    # print(test_net)
    test_input = torch.randn((2,6,256,256))
    fake_label = torch.from_numpy(np.array([0,1]))
    loss = nn.CrossEntropyLoss()
    testoutput = test_net(test_input)
    print(fake_label.shape)
    print(testoutput.shape)
    print(testoutput)
    print(loss(testoutput, fake_label))