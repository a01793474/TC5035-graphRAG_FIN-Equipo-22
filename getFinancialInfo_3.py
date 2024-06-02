import requests
import pandas as pd
from sec_edgar_api import EdgarClient
from time import sleep
import lxml.html
import os
import re
import math

user_agent="tec a01793474@tec.mx"
edgar = EdgarClient(user_agent=user_agent)


def get_file(acn, cik, user_agent,dirName,type,fileName,writeFile=False):
    texts = []
    numb = acn.replace('-','')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{numb}/{acn}.txt"
    r = requests.get(url, headers={"User-Agent":user_agent,
                               "Accept-Encoding":"gzip, deflate",
                               "Host":"www.sec.gov"})
    raw = r.text
    page = lxml.html.document_fromstring(raw)
    ret = page.cssselect('body')[0].text_content()
    text_tot = ret.translate(str.maketrans('', '', '\n\t\r')).lower()
    text_tot=re.sub('<.*?>', '', text_tot)
    text_tot=re.sub('[.*?]', '', text_tot)
    text_tot= re.sub(r'^(?=.{18})(?:[a-zA-Z]+\d|\d+[a-zA-Z])[a-zA-Z\d]*$', '', text_tot)
    combine_whitespace = re.compile(r"\s+")
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    text_tot=url_pattern.sub('', text_tot)
    text_tot=combine_whitespace.sub(' ', text_tot)
    
    while re.match("[<>#://()\]]",text_tot) != None:
        part_1_text = text_tot[0:re.match("[<>#://()\]]",text_tot).pos()-1]
        part_2_text = text_tot[re.match("[<>#://()\]]",text_tot).pos()+1:re.match("[<>#://()\]]",text_tot).end()]
        text_tot = part_1_text+part_2_text
    
          
    if writeFile:
        os.makedirs(dirName, exist_ok=True)
        with open(os.path.join(dirName, f'{fileName}.txt'), mode='w', newline='', encoding="utf-8") as file:
            file.write(text_tot)

    texts.append([f'{fileName}.txt',text_tot])
            
    return texts
  
def get_data(com_list,createFile=False):
  test = []
  headers = ["cik","name","entityType","sic","sicDescription",
            "tickers","exchanges","category","phone","address",
            "10-K-1"
            ]

  for i in range(0,len(com_list["cik"])):
    data = []
    cik = com_list["cik"][i]
    submission = edgar.get_submissions(cik=cik)
    data.append(submission['cik'])
    data.append(submission['name'])
    data.append(submission['entityType'])
    data.append(submission['sic'])
    data.append(submission['sicDescription'])
    data.append(submission['tickers'])
    data.append(submission['exchanges'])
    data.append(submission['category'])
    data.append(submission['phone'])
    
    data.append(submission['addresses']["business"]["street1"]+" "+
                submission['addresses']["business"]["city"]+" "+
                submission['addresses']["business"]["stateOrCountry"]+" "+
                submission['addresses']["business"]["zipCode"])
    
    subm = pd.DataFrame(submission["filings"]["recent"])
    
    k10 = subm[subm["form"] == "10-K"][0:1]
    # q10 = subm[subm["form"] == "10-Q"][0:1]
    
    for j in k10["accessionNumber"]:
        data.append([j,get_file(j, cik,user_agent,'sec_10K_data','10_K',f'10_K_{com_list["symbol"][i]}_{j}',createFile)])
        sleep(10)
    
    # for j in q10["accessionNumber"]:
    #     data.append([j,get_file(j, cik,user_agent,'data1','10_Q',f'10_Q_{com_list["symbol"][i]}_{j}',createFile)])
    #     sleep(10)
        
    test.append(data)
    

  test_pd = pd.DataFrame(test, columns=headers)
  
  return test_pd