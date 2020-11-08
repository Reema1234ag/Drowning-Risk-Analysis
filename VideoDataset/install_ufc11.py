from __future__ import print_function

from __future__ import print_function
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


from zipfile import ZipFile
import os


def download_and_extract(src):
    print ('Downloading ' + src)
    zip_file, h = urlretrieve(src, './delete.me')
    print ('Done downloading, start extracting.')
    try:
        with ZipFile(zip_file, 'r') as zfile:
            zfile.extractall('.')
            print ('Done extracting.')
    finally:
        os.remove(zip_file)

'''def generate_and_save_labels():
    groups = su.load_groups('./action_youtube_naudio')
    train, test = su.split_data(groups, '.avi')

    su.write_to_csv(train, os.path.join('.', 'train_map.csv'))
    su.write_to_csv(test, os.path.join('.', 'test_map.csv'))
'''
if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    download_and_extract('https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip')
    print ('Writing train and test CSV file...')
    #generate_and_save_labels()
    print ('Done.')