import ssl
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

import re
import torch
import random
import pandas as pd
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words_eng = set(stopwords.words('english'))
stop_words_chi = set(stopwords.words('chinese'))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    # transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)


def clean_title_chinese(text):
    pattern = re.compile(r'\([^)]*\)')
    text = re.sub(pattern, '', text)
    return text


def clean_text_chinese(text):
    characters_to_remove = ['\n', '\u3000', '<br/>']
    for char in characters_to_remove:
        text = text.replace(char, '')
    
    pattern = re.compile(r'\([^)]*\)')
    punctuation = re.compile(r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+')

    text = re.sub(pattern, '', text)
    text = re.sub(punctuation, '', text)

    # remove whitespaces 
    text = ' '.join(text.split()) 
    return text

def clean_text_english(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


def remove_stopwords(text, language):
    stop_words = stop_words_chi if language == 'chinese' else stop_words_eng
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def specialize(id, language):
    """
    Distinguish id of Chinese and English movies.
    """
    if language == 'chinese': return 'c' + str(id)
    return 'e' + str(id)


genre_map = {'剧情':'Drama', '喜剧':'Comedy', '动作':'Action', '爱情':'Romance', '科幻':'Science Fiction', 
             '动画':'Animation', '惊悚':'Thriller', '恐怖':'Horror', '纪录片':'Documentary', '犯罪':'Crime', 
             '冒险':'Adventure'}

def chinese_genre_transform(s):
    genres = s.split(',')
    output = []
    for genre in genres:
        if genre in genre_map.keys():
            output.append(genre_map[genre])
    return output

def english_genre_transform(s):
    return s.split(',')


def mian():
    chinese = pd.read_csv('data/chinese_movies.csv')
    english = pd.read_csv('data/english_movies.csv', engine='python')
    chinese.dropna(inplace=True)
    english.dropna(inplace=True)
    chinese['clean_title'] = chinese['title'].apply(clean_title_chinese)
    chinese['clean_overview'] = chinese['overview'].apply(clean_text_chinese)
    chinese['clean_overview'] = chinese['clean_overview'].apply(partial(remove_stopwords, language='chinese'))
    english['clean_title'] = english['title']
    english['clean_overview'] = english['overview'].apply(clean_text_english)
    english['clean_overview'] = english['clean_overview'].apply(partial(remove_stopwords, language='english'))
    chinese['id'] = chinese['id'].apply(partial(specialize, language='chinese'))
    english['id'] = english['id'].apply(partial(specialize, language='english'))
    chinese['genres_new'] = chinese['genres'].apply(chinese_genre_transform)
    english['genres_new'] = english['genres'].apply(english_genre_transform)
    df = pd.concat([english, chinese], ignore_index=True)
    df.drop('image', axis=1, inplace=True)

    mlb = MultiLabelBinarizer()
    one_hot_coding = mlb.fit_transform(df['genres_new'].tolist())
    res = pd.DataFrame(one_hot_coding, columns=mlb.classes_)
    concatenated_df = pd.concat([df, res], axis=1)

    shuffled_df = concatenated_df.sample(frac=1, random_state=595)
    shuffled_df.to_csv('movies.csv', columns=['id', 'clean_title', 'clean_overview', 'genres_new'] + mlb.classes_.tolist(), index=False) 


if __name__ == '__main__':
    mian()