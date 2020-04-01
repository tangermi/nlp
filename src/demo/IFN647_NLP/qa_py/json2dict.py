import json
from get_title import get_title_from_url

datadir = r"C:\Users\Computer\OneDrive - Queensland University of Technology\IFN647\LuceneBaselineSystem\data\collection.json"
with open(datadir) as json_file:
    data = json.load(json_file)


passages_collection = {}
passages_collection['passages'] = []

for query in data:
    for passage in query['passages']:
        url = str(passage['url'])
        passage_id = str(passage['passage_ID'])
        passage_text = str(passage['passage_text'])
        title = str(get_title_from_url(url))
        passages_collection['passages'].append({
            'id': passage_id,
            'url': url,
            'passage_text': passage_text,
            'title': title
        })

# get the first 10000 eletements for testing
passages_collection_10000 = {}
passages_collection_10000['passages'] = []
passages_collection_10000['passages'] = passages_collection['passages'][:10000]

# get the first 100000 eletements for testing
passages_collection_100000 = {}
passages_collection_100000['passages'] = []
passages_collection_100000['passages'] = passages_collection['passages'][:100000]

# test
# print(passages_collection['passages'][0])
