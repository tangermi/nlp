from search import search
from display_result import display

# the following search the expanded query text
# results, runtime = search('dance sing rap basketball')

# # query expansion can be turned off by adding the argument expansion=False
# results, runtime = search('dance sing rap basketball', expansion=False)

# # phrase searching can be applied by adding quotes to the query text, stopwords are ignored
# results, runtime = search("\"dance sing\"", expansion=False)

# the following matches if a document has sing within 5 words after dance, stopwords are ignored
results, runtime = search("\"dance sing\"~5", expansion=False)
display(results)