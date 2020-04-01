import eel
from search import search
import json

eel.init('web')

@eel.expose
def py_search(query_text):
    results = search(query_text)
    results_json = json.dumps(results)
    return results_json

eel.start('main.html')

