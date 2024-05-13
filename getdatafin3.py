import pandas as pd
import finnhub
from tqdm import tqdm
import yfinance as yf
import os
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
 
  start_date = (datetime.today() + relativedelta(months=-12)).strftime('%Y-%m-%d')
  end_date = datetime.today().strftime('%Y-%m-%d')
  
  for i in tqdm(range(0,len(com_list))):
    symbol = com_list['symbol'][i]
    cipk = com_list['cipk'][i]
    
    try:
      
      with open(f'./data/{symbol}_data.txt', mode='w',encoding="utf-8") as file:
        profile = finnhub_client.company_profile2(symbol=symbol)
        facts = edgar.get_company_facts(cik=cipk)
        data.append("[Company Introduction]:\n")
        file.write("[Company Introduction]:\n")
        
        company_template = "{name} is a leading entity in the {finnhubIndustry} sector. " \
                          "Incorporated and publicly traded since {ipo}, the company has established "\
                          "its reputation as one of the key players in the market. As of today, {name} "\
                          "has a market capitalization of {marketCapitalization:.2f} in {currency}, with "\
                          "{shareOutstanding:.2f} shares outstanding.\n"
        company_str = company_template.format(**profile)
        file.write(company_str)
        data.append(company_str)
        
        company_template = "{name} operates primarily in the {country}, trading under the ticker {ticker} "\
                           "on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company "\
                            "continues to innovate and drive progress within the industry.\n"
        company_str = company_template.format(**profile)
        file.write(company_str)
        data.append(company_str)
        
        submissions = edgar.get_submissions(cik=cipk)
        file.write("\n\n[Address]:\n")
        data.append("\n\n[Address]:\n")
        company_template = "Main office address is in {street1} {street2} {city},{stateOrCountry} {zipCode}\n"
        company_str = company_template.format(**submissions['addresses']['business'])
        file.write(company_str)
        data.append(company_str)
        company_template = "tel: {phone}\n"
        company_str = company_template.format(**submissions)
        file.write(company_str)
        data.append(company_str)
        
        file.write("\n\n[Stock]:\n")
        data.append("\n\n[Stock]:\n")
        company_template = "{label} this means {description}\n"
        fact_info=facts['facts']['dei']['EntityCommonStockSharesOutstanding']
        company_str = company_template.format(**fact_info)
        file.write(company_str)
        data.append(company_str)
    
        file.write("\n\n[News]:\n")
        data.append("\n\n[News]:\n")
        news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        for i in news:
          news_str = f"[Date]:{str(datetime.fromtimestamp(i['datetime']))}\n"
          file.write(news_str)
          data.append(news_str)
          
          news_template = "[Headline]:{headline}\n"
          news_str = news_template.format(**i)
          file.write(news_str)
          data.append(news_str)
          
          news_template = "[Summary]:{summary}\n\n"
          news_str = news_template.format(**i)
          file.write(news_str)
          data.append(news_str)
          
        file.write("\n\n[Fileings]:\n")
        data.append("\n\n[Fileings]:\n")
        facts_info = facts['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares']
        facts_list = []
        for i in facts_info:
            date = datetime.strptime(i['end'],'%Y-%m-%d')
            facts_list.append([date,date.strftime("%Y-%m-%d"), i['val'],i['accn'],i['fy'],i['fp'],i['form'],i['filed']])
            
        df = pd.DataFrame(facts_list, columns = ['end','date','val','accn','fy','fp','form','filed'])

        fact=df[df['end'] > datetime.strptime('2023-05-04','%Y-%m-%d')]
        fact.pop('end')
        fact.reset_index(inplace=True)
        
        for i in range(0,len(fact['date'])):
          
          fact_str = f"[Date]:{fact['date'][i]}\n"
          file.write(fact_str)
          data.append(fact_str)
          
          fact_str = f"[val]:{fact['val'][i]}\n"
          file.write(fact_str)
          data.append(fact_str)
          
          fact_str = f"[accn]:{fact['accn'][i]}\n"
          file.write(fact_str)
          data.append(fact_str)
          
          fact_str = f"[fy]:{fact['fy'][i]}\n"
          file.write(fact_str)
          data.append(fact_str)
          
          fact_str = f"[fp]:{fact['fp'][i]}\n"
          file.write(fact_str)
          data.append(fact_str)
          
          fact_str = f"[form]:{fact['form'][i]}\n\n"
          file.write(fact_str)
          data.append(fact_str)
          
        file.write("\n\n[Stock Movements]:\n")
        data.append("\n\n[Stock Movements]:\n")
        stock = yf.download(symbol, start=start_date, end=end_date).reset_index()
        
        for i in range(0,len(stock['Date'])):
          stock_str = f"[Date]:{stock['Date'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[Open]:{stock['Open'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[High]:{stock['High'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[Low]:{stock['Low'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[Close]:{stock['Close'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[Adj Close]:{stock['Adj Close'][i]}\n"
          file.write(stock_str)
          data.append(stock_str)
          
          stock_str = f"[Volume]:{stock['Volume'][i]}\n\n"
          file.write(stock_str)
          data.append(stock_str)
          
        time.sleep(8)
      
    except Exception as e:
      print(f" {e} in symbol {symbol}")
      
  return data