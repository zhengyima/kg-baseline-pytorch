# kg-baseline-pytorch
2019百度的关系抽取比赛，Pytorch版苏神的baseline，联合关系抽取

## 模型
与苏神的模型相同，只不过开发框架由Keras+Tensorflow变成了Pytorch，给使用Pytorch的小伙伴分享。

链接：https://github.com/bojone/kg-2019-baseline

`用BiLSTM做联合标注，先预测subject，然后根据suject同时预测object和predicate，标注结构是“半指针-半标注”结构，以前也曾介绍过（ https://kexue.fm/archives/5409 ）

标注结构是自己设计的，我看了很多关系抽取的论文，没有发现类似的做法。所以，如果你基于此模型做出后的修改，最终获奖了或者发表paper什么的，烦请注明一下（其实也不是太奢望）

```
@misc{
  jianlin2019bdkg,
  title={Hybrid Structure of Pointer and Ragging for Relation Extraction: A Baseline},
  author={Jianlin Su},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/bojone/kg-2019-baseline}},
}
```
`

## 用法
`python trans.py`转换数据，`python main.py`跑模型并观察结果。

## 结果
5个epoch到达0.73，最高能到0.75。

## 环境
Python 3.5+
Pytorch 1.0.1
tqdm




## 链接
- https://github.com/bojone/kg-2019-baseline
- https://pytorch.org/
