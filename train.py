"""
train.py
"""
# import copy
import torch


def train_model(model, loss_of_model, optimizer, scheduler, num_epochs, data_Loaders, device):
    """模型训练函数

    Args:
        model (_type_): 输入的模型
        loss_of_model (_type_): 损失函数
        optimizer (_type_): 优化器
        scheduler (_type_): 学习率策略
        num_epochs (_type_): 训练轮次
        data_Loaders (_type_): 数据批次加载器
        device (_type_): cpu/gpu模式
    Returns:
        model: 训练后的模型
        val_acc_history: 历史验证准确度
        train_acc_history: 历史训练准确度
        valid_loss: 历史验证损失
        train_loss: 历史训练损失
    """

    # 初始化变量
    model.to(device)  # 转移模型到GPU
    train_loss = []  # 训练损失
    valid_loss = []  # 验证损失
    val_acc_history = []  # 验证准确度
    train_acc_history = []  # 训练准确度
    best_accuracy = 0  # 最佳准确度
    # best_model_dict = copy.deepcopy(model.state_dict())  # 最佳模型

    # 开始训练
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)

        for stage in ['train', 'valid']:
            if stage == 'train':
                # 切换到训练模式
                model.train()
            elif stage == 'valid':
                # 切换到验证模式
                model.eval()

            running_loss = 0.0  # 当前轮次累计损失
            running_correct = 0  # 当前轮次累计正确个数

            for inputs, labels in data_Loaders[stage]:
                # 转移数据到GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 清零
                outputs = model(inputs)  # 输出
                loss = loss_of_model(outputs, labels)  # 计算损失
                _, pred = torch.max(outputs, 1)  # 预测类别
                if stage == 'train':
                    loss.backward()  # 训练阶段反向传播
                    optimizer.step()  # 训练阶段优化器更新参数
                running_loss = running_loss + loss.item()*inputs.size(0)  # 计算当前轮次累计损失
                running_correct = running_correct + \
                    torch.sum(pred == labels)  # 统计当前轮次累计正确个数

            epoch_loss = running_loss / \
                len(data_Loaders[stage].dataset)  # 计算当前轮次平均损失
            epoch_accuracy = running_correct.double(
            )/len(data_Loaders[stage].dataset)  # 计算当前轮次准确率

            print('Stage-{},Loss-{},Accuracy-{}'.format(stage,
                  epoch_loss, epoch_accuracy))  # 输出阶段，损失和准确率

            if stage == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy  # 更新最佳准确度
                # best_model_dict = copy.deepcopy(model.state_dict())  # 更新最佳模型
                # state = {
                #   'state_dict': best_model_dict,  # 最佳模型
                #  'best_accuracy': best_accuracy,  # 最佳准确率
                # 'optimizer': optimizer.state_dict()  # 优化器状态
                # }
                # torch.save(state, filename)  # 保存最佳模型，最佳准确率，优化器状态
            if stage == 'valid':
                val_acc_history.append(epoch_accuracy)  # 历史验证准确度
                valid_loss.append(epoch_loss)  # 历史验证损失
                # scheduler.step(epoch_loss)  # 学习率衰减
            if stage == 'train':
                train_acc_history.append(epoch_accuracy)  # 历史训练准确度
                train_loss.append(epoch_loss)  # 历史训练损失
        print('Optimizer learning rate:{:.7f}'.format(
            optimizer.param_groups[0]['lr']))  # 输出当前学习率
        print('best_accuracy:{:.7f}'.format(best_accuracy))
        scheduler.step()  # 学习率衰减
    return model, val_acc_history, train_acc_history, valid_loss, train_loss
