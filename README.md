更新时间: 2019/12/30    -tjw

## 内容结构
### 主目录
├── bin  
├── conf  
├── data  
├── source  
├── src  
└── README.md  

bin 用来存放运行程序的脚本  
conf 用来存放运行脚本的配置文件，包含各种运行参数  
data 用来存放临时测试的数据  
source 用来存放运行配置文件的逻辑  
src 用来存放各种代码，前面的配置文件决定了运行其中的哪一部分  


### 子目录
####  bin 目录
├── bin  
│   ├── run.sh  
│   └── run_test.sh  

run.sh 程序运行脚本，内容包含：数据处理，模型运行，以及模型评估  
run_test.sh 程序运行脚本，运行实际能演示的例子(直接使用训练好的模型)  

####  conf 目录
├── conf  
│   ├── classify  
│   ├── intention  
│   ├── regression  
│   ├── sentiment  
│   ├── similarity  
│   ├── test  
│   ├── __init__.py  
│   └── base.py  

classify 分类问题的配置文件  
intention 意图识别的配置文件  
regression 线性回归的配置文件  
sentiment 语义识别的配置文件  
similarity 相似性问题的配置文件  
test 测试用配置文件  
base.py 文件路径配置文件  

####  data 目录
├── data  
│   ├── similarity  

临时存放数据，随时可能删除，暂不做介绍  

####  source 目录
├── source  
│   ├── __init__.py  
│   └── config.py  

目前所有文件都根据config.py的逻辑运行  

####  src 目录
├── src  
│   ├── demo  
│   ├── evaluation  
│   ├── feature  
│   ├── generate  
│   ├── predict  
│   ├── predict_one  
│   ├── preprocess  
│   ├── similarity  
│   ├── text  
│   ├── train  
│   ├── utils  
│   ├── __init__.py  
│   ├── index.py  
│   └── test.py  

demo 做好的可运行demo，未加入系统  
evaluation 系统的一部分，用来做模型评估  
feature 系统的一部分，用来做特征处理  
generate 暂未使用  
predict 系统的一部分，使用训练好的模型对测试数据集进行预测，输出预测结果，服务之后的模型评估  
predict_one 系统的一部分，使用训练好的模型对用户输入的数据(可以是单条数据)进行预测，输出预测结果  
preprocess 系统的一部分，对数据进行读取以及预处理  
similarity 暂未使用  
text 暂未使用  
train 系统的一部分，构建模型，并使用处理好的数据/特征来训练模型  
utils 可复用的方法/功能/组件  
index.py 系统主体的运行逻辑  
test.py 系统测试的运行逻辑  


### 三级目录
#### demo 目录
├── src  
│   ├── demo  
│   │   ├── _fastNLP  
│   │   ├── _funNLP  
│   │   ├── _hanlp  
│   │   ├── _langid  
│   │   ├── _matplotlib  
│   │   ├── _numpy  
│   │   ├── _pandas  
│   │   ├── _pyltp  
│   │   ├── _scipy  
│   │   ├── _sklearn  
│   │   ├── _snowNLP  
│   │   ├── _stanfordNLP  
│   │   ├── _tensorflow  
│   │   ├── full_text_search  
│   │   ├── IFN647_NLP  
│   │   ├── intention  
│   │   ├── NLP_learn  
│   │   ├── NLPBook  
│   │   ├── sentiment  
│   │   ├── short_text  
│   │   ├── text  
│   │   ├── text_filter  
│   │   ├── wangfeng_LSTM  
│   │   ├── wulpus_world  
│   │   └── __init__.py  

_fastNLP 使用fastNLP作为工具，配合DPCNN神经网络对搜狗预料进行分本分类。尚未跑通，训练的第一个epoch就有bug，原因未知  
_funNLP 对github上funNLP里的2个工具进行封装测试，一个是cocoNLP基于正则的信息提取，另一个是命名实体识别  
_hanlp 对hanlp的各个功能进行封装以及测试，包括分词，文本分类，依存关系识别，文本摘要，word2vec等等  
_langid 对github上一个根据文本识别对应语种的工具，进行了修改及测试  
_matplotlib 用作测试  
_numpy numpy各种类别的练习题，总计100个  
_pandas pandas各种类别的练习，总计100个  
_pyltp 对pyltp进行封装及测试  
_scipy scipy里各种功能的测试  
_sklearn lihang的各种关于sklearn的demo  
_snowNLP snowNLP各种功能的demo，初看效果不如hanlp  
_stanfordNLP 斯坦福NLP的分词demo，模型较大，分词加载较慢  
_tensorflow tensorflow的各种机器学习demo，目前正在建设中  
full_text_search 一些whoosh的demo  
IFN647_NLP 一个文档搜索引擎的demo，分别使用c#的lucene.net和python的whoosh来实现  
intention 意图识别的demo  
NLP_learn 一本教材  
sentiment 语义识别的demo  
short_text 短文本聚类的demo  
text 各种文本分类的demo，包括:  
    1) _fastNLP 使用fastNLP作为工具，配合DPCNN神经网络对搜狗预料进行分本分类。尚未跑通，训练的第一个epoch就有bug，原因未知  
    2) _hanlp hanlp的文本分类  
    3) classifier 贝叶斯文本分类  
text_filter 本文过滤的demo  
wangfeng_LSTM 使用LSTM模型，以汪峰歌词作为语料进行训练，生成全新汪峰歌词的demo  
wulpus_world 通过计算概率分布，完成wulpus world游戏的demo  

#### evaluation 目录
├── src  
│   ├── evaluation  
│   │   ├── classify  
│   │   ├── image  
│   │   ├── rank  
│   │   ├── regression  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   └── evaluation.py  

classify 各种评价分类模型的指标/算法，包括f_score, jaccard, kappa, roc_auc和confuse_matrix  
image 目前只有评价siamese模型的指标/算法  
rank 暂未使用  
regression 评价回归模型的指标/算法，包括mean_square_error  
base.py  配置文件
evaluation.py 配置文件

#### feature 目录
├── src  
│   ├── feature  
│   │   ├── text  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   └── feature.py  

text multinomial_nb 和 xgboost 的特征处理  
base.py  配置文件  
feature.py 配置文件  

#### generate 目录
├── src  
│   ├── generate  
│   │   ├── __init__.py  
│   │   ├── ai_captcha.py  
│   │   ├── base.py  
│   │   └── generate.py  

ai_captcha.py  暂未使用  
base.py  配置文件  
generate.py 配置文件  

#### predict 目录
├── src  
│   ├── predict  
│   │   ├── classify  
│   │   ├── regression  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   └── predict.py  

classify  分类问题的测试集预测，包括multinomial_nb和xgboost模型  
regression 回归问题的测试集预测
base.py  配置文件  
generate.py 配置文件  

#### predict_one 目录
├── src  
│   ├── predict_one  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   ├── mpg.py  
│   │   ├── multinomial_nb.py  
│   │   ├── siamese.py  
│   │   ├── test_mpg.py  
│   │   ├── test_multinomial_nb.py  
│   │   ├── test_siamese.py  
│   │   ├── test_xgboost.py  
│   │   └── xgboost.py  

base.py  配置文件  
mpg.py 对于使用mpg数据训练好的线性回归模型，用户输入数据的测试  
multinomial_nb.py 对于训练好的multinomial_nb模型，用户输入数据的测试  
siamese.py 对于训练好的孪生神经网络模型，用户输入图片的测试  
test_mpg.py mpg测试的运行  
test_multinomial_nb.py nultinomial_nb测试的运行  
test_siamese siamese测试的运行  
test_xgboost xgboost测试的运行  
xgboost.py 对于训练好的xgboost模型，用户输入数据的测试  

#### preprocess 目录
├── src  
│   ├── preprocess  
│   │   ├── image  
│   │   ├── text  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   └── preprocess.py  

image 图片数据的预处理，包括mnist图片  
text  文字数据的预处理，包括sogou数据和mpg（实际上是数字数据）  
base.py  配置文件  
preprocess.py 配置文件  

#### train 目录
├── src  
│   ├── train  
│   │   ├── classify  
│   │   ├── regression  
│   │   ├── similarity  
│   │   ├── __init__.py  
│   │   ├── base.py  
│   │   └── train.py  

classify 分类器模型的训练，包括nultinomial_nb, xgboost  
regression 回归模型的训练，包括线性回归  
similarity 相似度模型的训练，目前有孪生神经网络  
base.py  配置文件  
train.py 配置文件  

#### utils 目录
├── src  
│   ├── utils  
│   │   ├── feature  
│   │   ├── preprocess  
│   │   ├── segment  
│   │   ├── similarity  
│   │   ├── __init__.py  
│   │   ├── load_data.py  
│   │   ├── logs.py  
│   │   ├── main.py  
│   │   ├── progress_bar.py  
│   │   └── tools.py  

feature 特征处理，包括tfidf/信息熵/卡方检查的计算  
preprocess 预处理，包括文档的读取与分词  
segment 封装的分词器，通过mode可以选用jieba/hanlp/pyltp进行分词  
similarity 相似度算法，包括siamese/simhash/余弦相似度/欧式距离...的计算  
load_data.py 文件读取的方法，目前没有被使用  
logs.py 日志处理  
similarity_demo.py 运行utils里的一个计算文本相似度的demo  
progress_bar.py 机器学习里自定义的一个进度条  
tools.py 各种工具，包括文件/模型的读取，softmax的实现，以及onehot的处理  
