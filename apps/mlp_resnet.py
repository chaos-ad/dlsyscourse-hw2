import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    result = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        nn.Sequential(*[
            ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob)
            for _ in range(num_blocks)
        ]),
        nn.Linear(hidden_dim, num_classes)
    )
    return result
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    total_loss = 0
    total_errors = 0
    total_batches = 0
    total_examples = 0
    loss_fn = ndl.nn.SoftmaxLoss()

    if opt:
        model.train()
    else:
        model.eval()

    for batch_idx, (X, y) in enumerate(dataloader):
        if opt:
            # print(f"TRAIN: {batch_idx=}, {X.shape=}, {y.shape=}")
            opt.reset_grad()
            y_prob = model(X)
            loss = loss_fn(y_prob, y)
            loss.backward()
            opt.step()
        else:
            # print(f"EVAL: {batch_idx=}, {X.shape=}, {y.shape=}")
            y_prob = model(X)
            loss = loss_fn(y_prob, y)

        y_prob = y_prob.numpy()
        y_pred = np.argmax(y_prob, axis=1)
        errors = np.not_equal(y_pred, y.numpy()).sum()

        total_loss += loss.numpy()
        total_errors += errors
        total_batches += 1
        total_examples += X.shape[0]

    # print(f"{total_loss=}, {total_errors=}, {total_batches=}, {total_examples=}")
    avg_loss = total_loss / total_batches
    avg_error_rate = total_errors / total_examples
    return (avg_error_rate, avg_loss)

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            f"{data_dir}/train-images-idx3-ubyte.gz",
            f"{data_dir}/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=True)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch_id in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        print(f"EPOCH {epoch_id}: {train_error=}, {train_loss=}, {test_error=}, {test_loss=}")

    return (train_error, train_loss, test_error, test_loss)


    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
