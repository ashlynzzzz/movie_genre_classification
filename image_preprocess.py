import os, shutil

def copyFile(fileDir, tarDir, s):
    pathDir = os.listdir(fileDir)
    if ".DS_Store" in pathDir:
        pathDir.remove(".DS_Store")
    for filename in pathDir:
        shutil.copyfile(fileDir+filename, tarDir+s+filename)

if __name__ == '__main__':
    tarDir = "movie_images/"
    os.makedirs(tarDir, exist_ok=True)
    copyFile("data/english_movie_images/", tarDir, 'e')
    copyFile("data/chinese_movie_images/", tarDir, 'c')