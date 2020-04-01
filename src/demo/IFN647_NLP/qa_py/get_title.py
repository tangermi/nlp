import re
import string
import requests
from contextlib import closing

def get_title_from_url(url):
    after_last_slash = url.rsplit('/', 1)[-1]
    res = re.sub('['+string.punctuation+']', ' ', after_last_slash).split() 
    title = ' '.join(res)
    return title

def download_title_from_url(url):
    try:
        from HTMLParser import HTMLParser
    except ImportError:
        from html.parser import HTMLParser
    CHUNKSIZE = 1024
    retitle = re.compile("<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
    buffer = ""
    htmlp = HTMLParser()
    with closing(requests.get(url, stream=True)) as res:
        for chunk in res.iter_content(chunk_size=CHUNKSIZE, decode_unicode=True):
            buffer = "".join([buffer, chunk])
            match = retitle.search(buffer)
            if match:
                title = htmlp.unescape(match.group(1))
                return title



#test
# print(get_title_from_url(r"https://en.wikipedia.org/wiki/Reserve_Bank_of_Australia"))
# print(download_title_from_url(r"https://en.wikipedia.org/wiki/Reserve_Bank_of_Australia"))
