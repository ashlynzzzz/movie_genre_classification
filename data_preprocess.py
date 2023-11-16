import pandas as pd
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer


def clean(x):
    """
    Eliminate redundant \n, space and <br/> from Chinese movie overviews.
    """
    characters_to_remove = ['\n', '\u3000', '<br/>']
    modified_string = x
    for char in characters_to_remove:
        modified_string = modified_string.replace(char, '')
    return modified_string

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
    chinese = pd.read_csv('chinese_movies.csv')
    english = pd.read_csv('english_movies.csv', engine='python')
    chinese.dropna(inplace=True)
    english.dropna(inplace=True)
    chinese['overview'] = chinese['overview'].apply(clean)
    chinese['id'] = chinese['id'].apply(partial(specialize, language='chinese'))
    english['id'] = english['id'].apply(partial(specialize, language='english'))
    chinese['genres'] = chinese['genres'].apply(chinese_genre_transform)
    english['genres'] = english['genres'].apply(english_genre_transform)
    df = pd.concat([english, chinese], ignore_index=True)
    df.drop('image', axis=1, inplace=True)

    mlb = MultiLabelBinarizer()
    one_hot_coding = mlb.fit_transform(df['genres'].tolist())
    res = pd.DataFrame(one_hot_coding, columns=mlb.classes_)
    concatenated_df = pd.concat([df, res], axis=1)

    shuffled_df = concatenated_df.sample(frac=1)
    shuffled_df.to_csv('movies.csv', index=False)


if __name__ == '__main__':
    mian()