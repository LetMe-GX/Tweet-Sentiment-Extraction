## 目前的进展

#### Results

Best Score: 0.716

Model: electra-base(2epoch-drop3 + 4epoch-drop5 + 5epoch-drop6 + only train on positive and negative data model ) + electra-large(2epoch-drop3) + albertxxlarge



Best two single model: 2epoch-drop3 + 4epoch-drop5



#### Have tried

1. using last two hidden layers (not help but not too bad)
2. Advasirial training(not help)
3. only train on positive and negative data(0.702, not help on single model but help on ensemble )
4. seperate train on positive and negative data(0.706) **need to try to ensemble**
5. add distance loss
6. k-fold ensemble (now useless on electra, but seem useful on roberta)



#### What we know

1. CV 和 LB的关系还未找到
2. 但从每个split的CV可以看出不同模型能够在不同部分数据上表现更好（证明了ensemble的有效性）
3. 数据中存在大量noise，目前不适宜做数据预处理和后处理
4. electra-base的表现，训练越久loss越低（每个epoch开始时loss会下降一点），甚至F1也在一直增长，但其实超过2epoch就基本overfitting了。不过，从CV分数看，主要是某几个split的CV在一直保持增长，某几个在2个epoch后就开始下降了。
5. Electra-large 对参数太敏感，很容易不收敛



## 下一步的计划

1. 数据方面

- 使用wordpiece处理[dataset](https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data)中所有数据(包含了private dataset中的数据), 将vocab合并进electra-base的vocab
- Fine-tune language model on this dataset ? (待定)



2. 模型结构

- Try [label smoothing](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/147070)
- 使用XLNET的输出层结构
- On the top of bert的网络结构：bilstm encoder + highway + bilstm decoder(CNN? Attention?)



3. ensemble

- Electra + roberta(很有提升的可能，但是需要将token转为word)，可[参考](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127551)
- Find new best single model 后重新ensemble
- 被ensemble的模型可以从这几个参数不同考虑epoch, sequence length, drop out



4. 工程层面（pipline）

- 替换掉目前代码中的evaluate，F1替换为jaccard
- 更好地实时获取evluate的数据
- 用sklearn的工具方法替换掉当前的split方案，保证完整的CV过程，最终输出所有CV值
- 保存所有有用的结果，log
- 可以兼容ensemble后的evaluate，用于测试ensemble后的CV值
- 可以便利输出各个模型ensemble后的CV值（穷举法？遗传算法？），以寻找最好的ensemble方案



5. 其他

- 特征工程
- 测试结果分析