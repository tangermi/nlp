import random
sentences = [('时间 问你 我们 群殴', '1'), ('大家 文献 二次 去啊', '0')]
segs= ['物品', '你的', '我的', '开心']
category = '0'
sentences.append((" ".join(segs), category))# 打标签
random.shuffle(sentences)
print(sentences)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word',  # tokenise by character ngrams
    max_features=4000,  # keep the most common 1000 ngrams
)
from sklearn.model_selection import train_test_split

#x是Content y是标签
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
print(x_train, x_test, y_train, y_test)

vec.fit(x_train)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
print(classifier.score(vec.transform(x_test), y_test))

pre = classifier.predict(vec.transform(x_test))
print(pre)