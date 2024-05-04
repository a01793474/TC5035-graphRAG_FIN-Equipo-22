import pandas as pd
import finnhub
from tqdm import tqdm
import yfinance as yf
import os
import json
import time
from datetime import datetime
from sec_edgar_api import EdgarClient
from dateutil.relativedelta import relativedelta

edgar = EdgarClient(user_agent="tec a01793474@tec.mx")
finnhub_client = finnhub.Client(api_key='coopla9r01qtvljfpqvgcoopla9r01qtvljfpr00')

def get_data(data_dir,com_list):
  """ Función para obtener y generar archivos json con la información"""
  data = []
  os.makedirs(f'{data_dir}', exist_ok=True) 
  os.makedirs(f'{data_dir}/news', exist_ok=True) 
  os.makedirs(f'{data_dir}/submissions', exist_ok=True) 
  os.makedirs(f'{data_dir}/facts', exist_ok=True) 
  os.makedirs(f'{data_dir}/stock', exist_ok=True)
  start_date = (datetime.today() + relativedelta(months=-12)).strftime('%Y-%m-%d')
  end_date = datetime.today().strftime('%Y-%m-%d')
  
  for i in tqdm(range(0,len(com_list)-1)):
    symbol = com_list['symbol'][i]
    cipk = com_list['cipk'][i]
    
    try:
      
      weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
      weekly_news = [{
                        "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                        "headline": n['headline'],
                        "summary": n['summary'],
                    } for n in weekly_news
                ]
      weekly_news.sort(key=lambda x: x['date'])
        
      json_file1 = f'{data_dir}/news/{symbol}_{start_date}_{end_date}_news.json' 
        
      with open(json_file1, mode='w', newline='',encoding="utf-8") as file1:
       file1.write(json.dumps({'news':weekly_news}))
        
      submissions = edgar.get_submissions(cik=cipk)
        
      json_file2 = f'{data_dir}/submissions/{symbol}_submissions.json' 
        
      with open(json_file2, mode='w', newline='',encoding="utf-8") as file2:
        file2.write(json.dumps({'submissions':submissions}))
              
      facts = edgar.get_company_facts(cik=cipk)
        
      json_file3 = f'{data_dir}/facts/{symbol}_facts.json' 
        
      with open(json_file3, mode='w', newline='',encoding="utf-8") as file3:
        file3.write(json.dumps({'facts':facts}))
        
      stock = yf.download(symbol, start=start_date, end=end_date).reset_index()
        
      json_file4 = f'{data_dir}/stock/{symbol}_stock.json' 
        
      stock.to_json(json_file4)
                  
      data.append({'symbol':symbol,
                      'data': {
                        'sumbissions': submissions,
                        'facts':facts,
                        'news':weekly_news
                      }
                  })
      
      time.sleep(1)
      
    except Exception as e:
      print(f" {e} in symbol {symbol}")
      
  return data

com_list = pd.read_csv('./data/ticker_select.txt', sep='\t',names=['symbol', 'cipk'])
data = get_data('data',com_list)
