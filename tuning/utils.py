import torch 
from tqdm import tqdm
from torch import nn


class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.lr = 1.5e-4
        self.criteria = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [80,140], gamma=0.1)
        self.loss_function = nn.BCELoss()

    
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            # 从batch中拿到训练数据
            inputs, targets = self.to_device(inputs), targets.to(self.device)
            # 传入模型进行前向传递
            outputs = self.model(inputs)
            # 计算损失
            loss = self.criteria(outputs.view(-1), targets.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += float(loss)
        
        # self.scheduler.step()
        return total_loss
    
    def test(self, test_loader, validation_dataset):
        self.model.eval()
        total_loss = 0.
        total_correct = 0
        for inputs, targets in test_loader:
            inputs, targets = self.to_device(inputs), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criteria(outputs.view(-1), targets.float())
            total_loss += float(loss)

            correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
            total_correct += correct_num

        return float(total_correct / len(validation_dataset)), total_loss / len(validation_dataset)



    def to_device(self, dict_tensors):
        result_tensors = {}
        for key, value in dict_tensors.items():
            result_tensors[key] = value.to(self.device)
        return result_tensors
        