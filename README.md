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

# Day 5：Baseline 与 DummyRegressor

第一次完成了一个完整的回归 Baseline 实验。

## 为什么需要 Baseline

在训练真正的机器学习模型之前，需要先建立一个简单的基准模型。

Baseline 用于回答：

> 一个真正的模型是否比最简单的预测方法更好？

如果真正的模型连 Baseline 都无法超过，就需要重新检查模型、Feature 或实验设计。

## DummyRegressor

本次使用：

`DummyRegressor(strategy="mean")`

该模型在训练时学习 `y_train` 的平均值，然后对所有测试样本预测同一个值。

当前训练集 G3 平均值约为：

`11.79`

因此 DummyRegressor 对不同的测试学生都预测约：

`11.79`

## fit 与 predict

`fit(X_train, y_train)`

用于让模型从训练数据中学习。

`predict(X_test)`

用于让已经训练完成的模型对没有参与训练的测试数据进行预测。

当前数据流：

X_train + y_train
↓
DummyRegressor.fit()
↓
训练完成的 Baseline
↓
X_test
↓
predict()
↓
y_pred_dummy

## RMSE

本次使用：

`root_mean_squared_error(y_test, y_pred_dummy)`

比较：

- `y_test`：测试集真实 G3
- `y_pred_dummy`：模型预测 G3

本次 Baseline 结果：

`RMSE ≈ 3.17`

可以初步理解为：

> 这个 Baseline 在当前测试集上的预测误差量级约为 3.17 个成绩分数单位。

RMSE 越小，说明预测结果整体越接近真实值。

RMSE = 0 表示预测完全正确。

需要注意：

RMSE ≈ 3.17 并不代表每一个学生都刚好预测错 3.17 分。

---

## Experiment 001

Problem:

Regression

Target:

G3

Features:

- age
- studytime
- failures
- absences

Excluded:

- G1
- G2
- G3 不作为 Feature

Train / Test:

- test_size = 0.2
- random_state = 42
- 519 training samples
- 130 testing samples

Preprocessing:

暂未进行额外预处理，仅使用数值 Feature。

Model:

DummyRegressor(strategy="mean")

Metric:

RMSE

Result:

RMSE ≈ 3.17

Observation:

DummyRegressor 学习训练集 G3 的平均值约 11.79，并对所有测试样本预测相同的值。

Conclusion:

当前实验建立了回归任务的 Baseline。

后续模型需要在相同测试集和评价指标下与该结果比较。

Limitations:

当前 Baseline 不根据不同学生的 Feature 产生不同预测，因此只能作为最简单的比较基准。