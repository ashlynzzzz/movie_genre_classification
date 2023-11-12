from lxml import etree
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import os

movie_df = pd.DataFrame(columns=['IMDb', 'title', 'genres', 'overview', 'image'])

movie_list_path = 'urls.txt'
count = 0
urls = [line.strip() for line in open(movie_list_path, 'r')]
num = len(urls)
for n in range(790, num):
    count += 1
    url = urls[n]
    url = url.strip()
    movie_info = []
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54'
    }
    page_text = requests.get(url,headers=headers).text
    soup = BeautifulSoup(page_text,'lxml')
    tree = etree.HTML(page_text)

    # IMDb
    imdb_re = re.compile(r'<span class="pl">IMDb:</span> (?P<imdb>.*?)<br>', re.S)
    imdb = imdb_re.findall(page_text)
    movie_info.append(''.join(imdb))

    # title
    li_list=tree.xpath('//*[@id="content"]/h1//text()')
    if len(li_list) == 0:
        black_movie = str(n+1) + ',' + url + '\n'
        print(n+1, url)
        with open('black_list.txt', 'a') as file:
            file.write(black_movie)
        continue
    name = li_list[1]
    movie_info.append(name.split()[0])

    # genres
    type = soup.find_all('span', property='v:genre')
    movie_type = []
    for i in type:
        movie_type.append(i.text)
    movie_info.append(','.join(movie_type))

    # overview
    sum_re = re.compile(r'<span property="v:summary" class="">(?P<summary>.*?)</span>',re.S)
    summary = sum_re.findall(page_text)
    summary = ''.join(summary).replace(' ','')
    if summary == '':
        sum_re = re.compile(r'<span class="all hidden">(?P<summary>.*?)</span>',re.S)
        summary = sum_re.findall(page_text)
        summary = ''.join(summary).replace(' ','')
    movie_info.append(summary)

    # image / poster
    pic_res = re.compile(r'<div id="mainpic" class="">.*? <img src="(?P<mainpic>.*?)".*? />', re.S)
    pic = pic_res.findall(page_text)
    movie_info.append(''.join(pic))

    movie_df = pd.concat([movie_df, pd.DataFrame([movie_info], columns=movie_df.columns)], ignore_index=True)
    if count % 10 == 0:
        print("Complete", count + 790)
        movie_df.to_csv('chinese_movies.csv', index=False)

movie_df.to_csv('chinese_movies.csv', index=False)