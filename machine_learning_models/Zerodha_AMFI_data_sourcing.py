from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os, pandas as pd
import requests


# Initialize environment
load_dotenv()
kite_api = os.getenv('kite_connect_api').strip()
all_instrument_data = os.getenv('all_instrument_data')
mf_instrument_data = os.getenv('mf_instrument_data')
hist_instrument_data = os.getenv('hist_instrument_data')
mf_hist_nav_data = os.getenv('mf_hist_nav_data')
motilal_hist_nav_data = os.getenv('motilal_hist_nav_data')
aditya_birla_sun_life_hist_nav_data = os.getenv('aditya_birla_sun_life_hist_nav_data')

# Source data from Zerodha
try:
    kc = KiteConnect(api_key=kite_api)
    with open('/home/anurag_sarangi/projects/statistical_and_machine_learning_models/machine_learning_models/kite_access_token.txt', 'r') as f_kc_access_token:
        kc.set_access_token(f_kc_access_token.read().strip())

    df_instruments = pd.DataFrame(kc.mf_instruments())
    df_instruments.index.name = 'SNo.'
    df_instruments.to_csv(all_instrument_data)
    df_mf_instruments = pd.DataFrame(kc.mf_instruments())
    df_mf_instruments.index.name = 'SNo.'
    df_mf_instruments.to_csv(mf_instrument_data)
    
    # Get instruments for historical data
    instruments = kc.instruments('NSE')
    df_nse_instruments = pd.DataFrame(instruments)
    filtered_nse_instruments = df_nse_instruments.loc[df_nse_instruments['tradingsymbol']=='TCS', 'instrument_token']
    for values in filtered_nse_instruments:
        hist_data = pd.DataFrame(kc.historical_data(instrument_token=int(values), 
                                                    from_date='2025-01-01 00:00:00', 
                                                    to_date='2025-10-23 00:00:00', 
                                                    interval='day'))

    hist_data.to_csv(hist_instrument_data)

# Source data from AMFI Community Mirror
    # Pull relevant scheme code from API
    search_url = 'https://api.mfapi.in/mf' # community maintained mirror of official Association of Mutual Funds of India (AMFI) data
    resp = requests.get(search_url)
    out = resp.json()
    fund_name = input('Enter fund name: ')
    for funds in out:
        if fund_name in funds['schemeName']:
            scheme_code = funds['schemeCode']
        else:
            pass
    
    # Pull NAV data
    nav_data_url = f'https://api.mfapi.in/mf/{scheme_code}'
    resp_nav = requests.get(nav_data_url)
    nav_out = resp_nav.json()
    fund_house = nav_out['meta'].get('fund_house')
    df_nav_data = pd.DataFrame(nav_out['data'])
    df_nav_data.index.name = 'SNo.'
    df_nav_data['fund_name'] = fund_house
    df_nav_data.to_csv(f'{mf_hist_nav_data}/00.{fund_house}_Historical_NAV.csv')
except Exception as e:
    print(f'Error Encountered: {e}')