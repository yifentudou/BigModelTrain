import torch
from tqdm import tqdm


def get_loss_fn():
    return torch.nn.BCELoss()


def train(n_epochs, train_epoch_steps, test_epoch_steps, train_iter, test_iter,
          model, optimizer, loss_fn, loss_history, model_save_period, device):
    for epoch in range(n_epochs):
        print(f"********************开始第{epoch + 1}次迭代********************")
        model = model.train()
        pbar = tqdm(total=train_epoch_steps, desc=f'Train Epoch {epoch + 1}/{n_epochs}', postfix=dict, mininterval=0.3)
        train_loss_sum, train_acc_sum, train_n, train_batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.reshape(y.shape[0], 1).type(torch.FloatTensor).to(device)
            # 预测
            y_hat = model(X)
            # 计算损失
            loss = loss_fn(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += ((y_hat > 0.5).type(torch.int32) == y).cpu().sum().item()
            train_n += y.shape[0]
            train_batch_count += 1

            pbar.set_postfix(**{"train_loss": train_loss_sum / train_batch_count,
                                "train_acc": train_acc_sum / train_n})
            pbar.update(1)
        pbar.close()
        print("Train Epoch %d/%d, train_loss %.4f, train_acc %.4f" %
              (epoch + 1, n_epochs, train_loss_sum / train_batch_count, train_acc_sum / train_n))

        # 测试
        model = model.eval()
        pbar = tqdm(total=test_epoch_steps, desc=f'Test Epoch {epoch + 1}/{n_epochs}', postfix=dict, mininterval=0.3)
        test_loss_sum, test_acc_sum, test_n, test_batch_count = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                y = y.reshape(y.shape[0], 1).type(torch.FloatTensor).to(device)
                # 预测
                y_hat = model(X)
                # 计算损失
                loss = loss_fn(y_hat, y)

                test_loss_sum += loss.cpu().item()
                test_acc_sum += ((y_hat > 0.5).type(torch.int32) == y).cpu().sum().item()
                test_n += y.shape[0]
                test_batch_count += 1

                pbar.set_postfix(**{"test_loss": test_loss_sum / test_batch_count,
                                    "test_acc": test_acc_sum / test_n})
                pbar.update(1)
        pbar.close()
        print("Test Epoch %d/%d, test_loss %.4f, test_acc %.4f" %
              (epoch + 1, n_epochs, test_loss_sum / test_batch_count, test_acc_sum / test_n))

        loss_history.append_loss(train_loss_sum / train_batch_count,
                                 test_loss_sum / test_batch_count,
                                 train_acc_sum / train_n,
                                 test_acc_sum / test_n)

        if (epoch + 1) % model_save_period == 0:
            loss_history.save_model(model, epoch + 1,
                                    train_loss_sum / train_batch_count,
                                    test_loss_sum / test_batch_count,
                                    train_acc_sum / train_n,
                                    test_acc_sum / test_n
                                    )
