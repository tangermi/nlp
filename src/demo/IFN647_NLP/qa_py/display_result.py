def display(results):
    for passage in results:
        print(passage["title"])
        print(passage['content'])