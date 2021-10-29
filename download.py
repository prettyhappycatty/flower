# ライブラリのインポート
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys
 
# APIキーの情報
 
key = "6d9a0276610da3123de2f62e37a82523"
secret = "73c84758f3964722"
wait_time = 1
 
# 保存フォルダの指定
keyword = sys.argv[1]
savedir = "./data/" + keyword
 
# 接続クライアントの作成とサーチの実行
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = keyword,
    per_page = 500,
    media = 'photos',
    #sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, license'
)
 
# 結果の取り出しと格納
#print(result)
photos = result['photos']
print(len(photos))
for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)