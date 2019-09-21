from bs4 import BeautifulSoup
from bs4.element import Comment
import sqlite3
from common import (dir_loc,
                    db_name,
                    sep_char)
import csv



def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def get_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u". ".join(t.strip() for t in visible_texts)


def get_meta_info_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    metas = soup.find_all('meta')
    return '. '.join([meta.attrs['content'] for meta in metas if 'content' in meta.attrs])


def extract_useful_text_from_webpage():
    with sqlite3.connect(f'{dir_loc}/{db_name}') as conn_disk:
        res = list(conn_disk.execute('''select url, file_name from websites'''))

        for webpage in res:
            url = webpage[0]
            file_name = webpage[1]
            html = None

            try:
                with open(f'{dir_loc}/all_html_chunks/{file_name}.txt', 'r') as f:
                    # print(f'file {file_name} found')
                    lines = f.readlines()
                    for row in lines[1:]:
                        row_split = row.split('|')
                        if len(row_split) == 2 and row_split[0] == url:
                            html = row_split[1]
                            break
            except FileNotFoundError:
                print(f'file {file_name} not found')

            if html:
                page_text = get_meta_info_from_html(html)
                print(url, page_text)

if __name__ == '__main__':
    extract_useful_text_from_webpage()
