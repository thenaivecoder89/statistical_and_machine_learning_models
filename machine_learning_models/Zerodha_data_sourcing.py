from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os, pandas as pd


# Initialize environment
load_dotenv()
kite_api = os.getenv('kite_connect_api').strip()
all_instrument_data = os.getenv('all_instrument_data')
mf_instrument_data = os.getenv('mf_instrument_data')
hist_instrument_data = os.getenv('hist_instrument_data')

# Get instrument details
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
except Exception as e:
    print(f'Error Encountered: {e}')