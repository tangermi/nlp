from stanfordcorenlp import StanfordCoreNLP

if __name__ == 'main':
    nlp = StanfordCoreNLP(r'/home/xiaoxinwei/data/stanford-corenlp/')

    sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    print ('Tokenize:', nlp.word_tokenize(sentence))
    print ('Part of Speech:', nlp.pos_tag(sentence))
    print ('Named Entities:', nlp.ner(sentence))
    print ('Constituency Parsing:', nlp.parse(sentence))
    print ('Dependency Parsing:', nlp.dependency_parse(sentence))

    nlp.close() # Do not forget to close! The backend server will consume a lot memery.


    sentence = '清华大学位于北京。'

    with StanfordCoreNLP(r'/home/xiaoxinwei/data/stanford-corenlp/', lang='zh') as nlp:
        print(nlp.word_tokenize(sentence))
        print(nlp.pos_tag(sentence))
        print(nlp.ner(sentence))
        print(nlp.parse(sentence))
        print(nlp.dependency_parse(sentence))

