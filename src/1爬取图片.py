import os.path
import os,re

import requests
from lxml import etree
from tqdm import tqdm


def get_flo(URL,flower_path,flower_class:str):
    if not os.path.exists(os.path.join(flower_path,flower_class)):
        os.makedirs(os.path.join(flower_path,flower_class))
    header={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0'
    }
    pic_adress =[]
    for i in tqdm(range(1,401,10)):
        url = re.sub('first=1',f'first={i}',URL)
        res=requests.get(url,headers=header)
        f_adress=etree.HTML(res.text)
        p_adress = f_adress.xpath('//*[@id="mmComponent_images_1"]/ul/li/div/div/a/@m')
        p_adre = [eval(p)['turl'] for p in p_adress]
        pic_adress.extend(p_adre)
        # 去重
    set_pic = list(set(pic_adress))
    # 逐个解析并写出到文件
    for i in tqdm(range(len(set_pic))):
        # 解析
        ct = requests.get(set_pic[i]).content
        # 写出
        with open(flower_path + f'{flower_class}/{flower_class}_{i}.jpg', mode='wb') as p:
            p.write(ct)
urls = [
    'https://cn.bing.com/images/search?q=%e7%99%bd%e8%89%b2%e9%b8%a1%e8%9b%8b%e8%8a%b1&form=HDRSC2&first=1', # 白色鸡蛋花
    'https://cn.bing.com/images/search?q=%e8%92%b2%e5%85%ac%e8%8b%b1&form=HDRSC2&first=1', # 蒲公英
    'https://cn.bing.com/images/search?q=%e5%90%91%e6%97%a5%e8%91%b5&form=HDRSC2&first=1', # 向日葵
    'https://cn.bing.com/images/search?q=%e7%89%a1%e4%b8%b9&form=HDRSC2&first=1',# 牡丹
    'https://cn.bing.com/images/search?q=%e7%89%b5%e7%89%9b%e8%8a%b1&form=HDRSC2&first=1',# 牵牛花
]# 白色鸡蛋花frangipani、蒲公英dandelion、向日葵sunflower、牡丹peony、牵牛花morning_glory
flower_class = ['frangipani','dandelion', 'sunflower', 'peony', 'morning_glory']
for url, f_cla in zip(urls, flower_class):
    get_flo(url, flower_path='D:/work_data/2408flower/', flower_class=f_cla)