# Day 4：Train / Test Split

学习了为什么机器学习实验需要将数据划分为训练集和测试集。

## 为什么需要训练集和测试集

如果使用全部数据训练模型，再使用同一批数据评价模型，只能说明模型对已经见过的数据表现如何，无法可靠评价模型面对新数据时的预测能力。

因此需要将数据划分为：

* Training Set：用于模型学习
* Testing Set：用于最终评价模型

当前数据流：

```text
X / y
↓
train_test_split
↓
X_train / y_train
X_test  / y_test
```

其中：

* `X_train`：训练时提供给模型的 Feature
* `y_train`：训练时对应的真实 G3
* `X_test`：训练结束后用于预测的 Feature
* `y_test`：用于和模型预测结果比较的真实 G3

## 当前数据划分

使用：

```text
test_size = 0.2
random_state = 42
```

划分结果：

```text
X_train.shape = (519, 4)
X_test.shape  = (130, 4)

y_train.shape = (519,)
y_test.shape  = (130,)
```

共有：

* 519 个训练样本
* 130 个测试样本
* 总计 649 个样本

训练集和测试集中的 X 与 y 数量能够一一对应。

## test_size

`test_size=0.2` 表示大约使用 20% 的数据作为测试集，剩余约 80% 作为训练集。

## random_state

`random_state` 用于固定随机划分方式，使实验可以复现。

它决定：

“数据如何随机划分”

而不是：

“训练集和测试集分别有多少数据”。

在 `test_size` 不变的情况下，修改 `random_state` 通常不会改变训练集和测试集的数量，但可能改变具体哪些样本进入训练集和测试集。

## 当前原则

测试集用于模拟模型没有见过的新数据。

因此：

**测试数据不能参与模型的训练过程。**
