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

# Day 6：LinearRegression

第一次训练并评价真正利用 Feature 进行预测的回归模型：

`LinearRegression`

## DummyRegressor 与 LinearRegression

之前的 Baseline：

`DummyRegressor(strategy="mean")`

主要学习训练集 `y_train` 的平均 G3，并对所有测试样本预测相同的值。

LinearRegression 则会利用当前的 Feature：

* age
* studytime
* failures
* absences

学习这些 Feature 与 G3 之间的线性关系。

## 模型训练

使用：

`linear_model.fit(X_train, y_train)`

让 LinearRegression 根据训练数据学习模型参数。

训练完成后，模型学习到了：

* `coef_`：各 Feature 对线性预测公式的权重
* `intercept_`：线性预测公式中的基础值

当前学习到的系数约为：

* age：+0.0067
* studytime：+0.7648
* failures：-2.0302
* absences：-0.0190

intercept 约为：

`10.7554`

这些系数表示当前模型中的统计关系，不能直接解释为现实中的因果关系。

## 模型预测

使用：

`linear_model.predict(X_test)`

让训练完成的模型对测试集 Feature 进行预测。

与 DummyRegressor 不同，LinearRegression 会根据不同学生的 Feature 产生不同的预测 G3。

回归模型可以产生小数预测值，例如：

`12.33`

即使原始 G3 是整数，这也是正常现象。

## RMSE

LinearRegression 在当前测试集上的结果：

`RMSE ≈ 2.88`

之前的 Baseline：

`DummyRegressor RMSE ≈ 3.17`

当前比较：

| Model            | RMSE |
| ---------------- | ---: |
| DummyRegressor   | 3.17 |
| LinearRegression | 2.88 |

RMSE 越小越好。

因此，在保持：

* 相同 Features
* 相同训练集
* 相同测试集
* 相同评价指标

的情况下，LinearRegression 当前测试集上的预测误差低于 DummyRegressor。

当前只能得出：

> LinearRegression 在当前实验条件下优于 Baseline。

不能直接得出：

* LinearRegression 已经是一个非常准确的成绩预测模型
* Feature 与 G3 之间存在因果关系
* 当前模型可以直接用于真实学校决策

---

## Experiment 002

Problem:

Regression

Target:

G3

Features:

* age
* studytime
* failures
* absences

Excluded:

* G1
* G2
* G3 不作为 Feature

Train / Test:

* test_size = 0.2
* random_state = 42
* 519 training samples
* 130 testing samples

Preprocessing:

暂未进行额外预处理，仅使用数值 Feature。

Model:

LinearRegression

Metric:

RMSE

Result:

`RMSE ≈ 2.88`

Baseline:

`DummyRegressor RMSE ≈ 3.17`

Observation:

LinearRegression 会利用不同 Feature 学习不同权重，并针对不同测试样本产生不同预测结果。

Conclusion:

在当前实验设置和相同测试集下，LinearRegression 的 RMSE 低于 DummyRegressor，因此当前表现优于 Baseline。

Limitations:

当前只使用 4 个简单数值 Feature，并且只进行了一次 train/test split。

模型中的 Feature 系数表示当前数据和模型中的统计关系，不能直接解释为因果关系。

# Day 7：第一周总结

第一周完成了一个最小的机器学习回归实验流程。

## 完整实验流程

本项目目前的流程为：

CSV 数据
→ pandas 读取为 DataFrame
→ 数据理解与 EDA
→ 选择 Feature 和 Target
→ Train / Test Split
→ Baseline
→ LinearRegression
→ Predict
→ RMSE
→ 模型比较
→ 实验结论

## 数据与目标

当前任务属于 Regression（回归）。

Target：

- G3

第一版 Features：

- age
- studytime
- failures
- absences

当前暂不使用 G1 和 G2。

G3 本身不能作为 Feature，因为 G3 就是当前需要预测的 Target。如果将 G3 同时放入 X 和 y，相当于使用答案预测答案，会造成数据泄漏，使实验结果失去实际意义。

## Train / Test Split

数据共 649 条。

当前划分：

- Training samples：519
- Testing samples：130
- test_size = 0.2
- random_state = 42

训练阶段：

X_train + y_train
→ fit()

测试阶段：

X_test
→ predict()
→ y_pred

其中：

- X_train：训练用 Feature
- y_train：训练阶段对应的真实 G3
- X_test：模型测试时看到的 Feature
- y_pred：模型根据 X_test 得到的预测结果
- y_test：测试数据对应的真实 G3，用于评价模型

模型在 predict() 时不会看到 y_test。

## Baseline

使用：

DummyRegressor(strategy="mean")

作为最简单的 Baseline。

结果：

RMSE ≈ 3.17

Baseline 的主要作用是为后续模型提供一个参考标准。

如果一个更复杂的模型连 Baseline 都无法超过，就需要进一步分析模型、Feature、数据划分或其他实验设置。

## LinearRegression

第一个真正利用 Feature 进行学习的模型：

LinearRegression

模型通过：

fit(X_train, y_train)

学习 Feature 与 G3 之间的线性关系。

训练后可以得到：

- coef_
- intercept_

模型再通过：

predict(X_test)

得到预测结果 y_pred。

当前结果：

RMSE ≈ 2.88

## 模型比较

| Model | RMSE |
| --- | ---: |
| DummyRegressor | 3.17 |
| LinearRegression | 2.88 |

RMSE 越低表示当前测试集上的预测误差越小。

因此，在相同 Feature、相同 Train / Test Split 和相同评价指标下：

LinearRegression 当前测试集上的预测误差低于 DummyRegressor。

所以目前可以认为：

> LinearRegression 在本次实验条件下优于 Baseline。

但不能直接认为当前模型已经非常准确。

## 当前限制

目前实验仍然比较简单：

- 只使用 4 个数值 Feature
- 暂未使用分类 Feature
- 只进行了一次 Train / Test Split
- 暂未加入 G1 和 G2
- 当前实验结果不能直接推广到真实学校环境
- 模型中的统计关系不能直接解释为因果关系

## 第一周收获

第一周重点不是学习复杂模型，而是建立完整的机器学习实验思维：

先明确问题和数据，再选择 Feature 和 Target，划分训练集与测试集，建立 Baseline，训练模型，用未参与训练的数据评价模型，并根据指标进行有限而谨慎的实验结论。

当前已经完成第一个完整的 Regression 实验闭环。