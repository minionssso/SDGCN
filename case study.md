下面给出一个具体例子，解释我们是如何做可视化的：

1. 先在数据集上训练好一个模型M。
2. 将句子"The food is good" 输入M， 模型输出情感极性的分布(积极，消极，中立)[0.7, 0.1, 0.2]，因此预测的情感极性为积极。
3. 依次将句子中的词替换成"[pad]"符号，再次输入M，观察情感极性在积极上的相对变化量，将这个变化量视为该词对于极性预测的贡献量，我们认为变化量越大，说明该词对极性预测的贡献越大，即模型认为该词越重要，反之越小。
4. 例如将 "[pad] food is good" 输入M，模型输出的情感极性分布为[0.7, 0.1, 0.2]，则"The"这个词的贡献量为0 (0.7-0.7=0)。
5. 例如将"The food is [pad]" 输入M，模型输出的情感极性分布为[0.1, 0.1, 0.8]，则"good"这个词贡献量为0.6 (0.7-0.1=0.6)。
6. 最后我们跟据每个词的贡献量，用heatmap画出可视化图，贡献量越大的词，颜色越深。

可视化代码已上传，请查看heapmap.py 和 mask_exp.py