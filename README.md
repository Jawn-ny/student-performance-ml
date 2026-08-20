# Student Performance ML

学生成绩数据分析与机器学习实验项目。

本项目使用 Python 进行数据分析，并逐步学习完整的机器学习实验流程，包括数据理解、探索性数据分析、特征与目标、训练集与测试集、Baseline、模型训练、评价指标和错误分析等。

## 数据来源

数据来自 UCI Machine Learning Repository：

**Student Performance Dataset**

当前使用的数据文件：

`student-por.csv`

该文件记录了 Portuguese language 课程的学生数据。

## Day 1：认识数据

已完成：

* 使用 pandas 读取 `student-por.csv`
* 了解 Dataset、Sample、Feature、Target 和 DataFrame
* 查看数据前几行
* 查看 DataFrame 的行数和列数
* 查看字段名称
* 查看字段的数据类型
* 查看数值字段的基本统计信息
* 检查缺失值
* 初步查看目标字段 `G3`

当前数据规模：

* 649 条记录
* 33 个字段

缺失值检查结果：

* 当前数据中未发现缺失值

当前回归任务计划预测的目标：

`G3` —— 学生最终成绩。
