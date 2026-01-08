import os
from dotenv import load_dotenv
from datetime import datetime
from io import StringIO
from kiteconnect import KiteConnect
import cudf, traceback, time, json, cupy, yfinance as yf

# Main class comprising of multiple methods for session initialization and ETL operations.
class ZerodhaDataPipeline:

    def __init__(self, api, token_path, api_secret):
        self.api = api
        self.token_path = token_path
        self.api_secret = api_secret

    def initialize_kite_session(self):
        kc = KiteConnect(api_key=self.api)
        print(f'Login URL: {kc.login_url()}')
        request_token = input(f'Enter request token for the day: ').strip()
        data = kc.generate_session(request_token=request_token, api_secret=self.api_secret)
        with open(self.token_path, 'w') as file:
            json.dump({
                'access_token': data['access_token'],
                'date': datetime.now().strftime('%Y-%m-%d')
            }, file)
        print('File generated successfully.')
    
    def data_extract_instruments_master(self):
        kc = KiteConnect(api_key=self.api)
        instruments_data = {
            'tradingsymbol': [data['tradingsymbol'] for data in kc.instruments()],
            'instrument_token': [data['instrument_token'] for data in kc.instruments()],
            'exchange': [data['exchange'] for data in kc.instruments()],
            'segment': [data['segment'] for data in kc.instruments()],
            'name': [data['name'] for data in kc.instruments()]
        }
        cudf_instruments = cudf.DataFrame(instruments_data)
        return cudf_instruments
    
    def historical_data_extract(self, instrument_token, from_date, to_date, interval):
        kc = KiteConnect(api_key=self.api)
        with open(self.token_path, 'r') as file:
            access_token_json = json.load(fp=file)
            access_token = access_token_json['access_token']
        kc.set_access_token(access_token)
        data_dump = kc.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False,
            oi=False
        )
        historical_data = {
            'date': [data['date'].date().isoformat() for data in data_dump],
            'closing_value': [float(data['close']) for data in data_dump]
        }
        historical_json = json.dumps(historical_data)
        return historical_json

# Main program codebase - comprises of user menu options and calls to corresponding methods.
def main_program():
    load_dotenv()
    kite_connect_api = os.getenv('kite_connect_api')
    kite_connect_secret = os.getenv('kite_connect_secret')
    kite_instruments_data = os.getenv('kite_instruments_data')
    kite_access_token_path = os.getenv('kite_access_token_path')
    stored_instruments_df = cudf.read_csv(kite_instruments_data)
    kite_nifty50_daily_historical_data = os.getenv('kite_nifty50_daily_historical_data')
    kite_niftybank_daily_historical_data = os.getenv('kite_niftybank_daily_historical_data')
    gold_daily_historical_data = os.getenv('gold_daily_historical_data')
    pipeline = ZerodhaDataPipeline(api=kite_connect_api, 
                                   token_path=kite_access_token_path, 
                                   api_secret=kite_connect_secret)
    select_option = int(input("""
        Available Options:
        0-Initialize_Kite_Session: **Run this once at the start before running any other option**
        1-Data_Extract_Instruments_Master
        2-Historical_Data_Extract_and_Log_Returns_Calculation_NIFTY50
        3-Historical_Data_Extract_and_Log_Returns_Calculation_NIFTYBank
        4-Historical_Data_Extract_and_Log_Returns_Calculation_Gold_Proxy
        
        Select Option: """))
    # Initialize kite session for the day
    if select_option == 0:
        pipeline.initialize_kite_session()
    # Codebase for extracting all instruments from Kite
    elif select_option == 1:
        start_time = time.perf_counter()
        instruments_df = pipeline.data_extract_instruments_master()
        instruments_df.index.name = 'SNo.'
        instruments_df.to_csv(kite_instruments_data)
        print(f'Loaded {len(instruments_df)} records in csv file.')
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'Total program runtime: {runtime:.2f} seconds.')
    # Codebase for extracting and transforming (1st round transformation) historical data for NIFTY50
    elif select_option == 2:
        start_time = time.perf_counter()
        
        # Data extract
        json_final = {
            'date': [],
            'closing_value': []
        }
        filtered_instruments_df = stored_instruments_df[
            stored_instruments_df['tradingsymbol'] == 'NIFTY 50'
        ]
        instrument_token = filtered_instruments_df['instrument_token'].iloc[0]
        print(f'Instrument Token: {instrument_token}')
        dates_range = ['2001-01-01', '2006-01-01', '2011-01-01', '2016-01-01', '2021-01-01', '2026-01-01']
        interval = 'day'
        for d in range(len(dates_range)-1):
            json_str = pipeline.historical_data_extract(instrument_token, 
                                                        dates_range[d], 
                                                        dates_range[d+1], 
                                                        interval)
            json_data = json.loads(json_str)
            json_final['date'].extend(json_data['date'])
            json_final['closing_value'].extend(json_data['closing_value'])
        
        json_final_str = json.dumps(json_final)
        nifty50_df = cudf.read_json(StringIO(json_final_str))
        
        # Data transformations
        nifty50_df.index.name = 'SNo.'
        nifty50_df['date'] = nifty50_df['date'].dt.strftime('%Y-%m-%d')
        nifty50_df['daily_closing_pct_change'] = nifty50_df['closing_value'].pct_change()
        nifty50_df = nifty50_df.dropna(
            subset='daily_closing_pct_change',
            how='all'
        )
        nifty50_df['daily_log_closing_value'] = cupy.log(1 + nifty50_df['daily_closing_pct_change'])
        nifty50_df['instrument_token'] = instrument_token
        nifty50_df['trading_symbol'] = 'NIFTY 50'
        print(f'Top 5 rows NIFTY50 Historical Data (2001-2026):\n{nifty50_df.head()}')
        print(f'Last 5 rows NIFTY50 Historical Data (2001-2026):\n{nifty50_df.tail()}')
        
        nifty50_df.to_csv(kite_nifty50_daily_historical_data)
        print(f'Loaded {len(nifty50_df)} records in csv file.')
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'Total program runtime: {runtime:.2f} seconds.')
    # Codebase for extracting and transforming (1st round transformation) historical data for NIFTY BANK
    elif select_option == 3:
        start_time = time.perf_counter()
        
        # Data extract
        json_final = {
            'date': [],
            'closing_value': []
        }
        filtered_instruments_df = stored_instruments_df[
            stored_instruments_df['tradingsymbol'] == 'NIFTY BANK'
        ]
        instrument_token = filtered_instruments_df['instrument_token'].iloc[0]
        print(f'Instrument Token: {instrument_token}')
        dates_range = ['2001-01-01', '2006-01-01', '2011-01-01', '2016-01-01', '2021-01-01', '2026-01-01']
        interval = 'day'
        for d in range(len(dates_range)-1):
            json_str = pipeline.historical_data_extract(instrument_token, 
                                                        dates_range[d], 
                                                        dates_range[d+1], 
                                                        interval)
            json_data = json.loads(json_str)
            json_final['date'].extend(json_data['date'])
            json_final['closing_value'].extend(json_data['closing_value'])
        
        json_final_str = json.dumps(json_final)
        niftybank_df = cudf.read_json(StringIO(json_final_str))
        
        # Data transformations
        niftybank_df.index.name = 'SNo.'
        niftybank_df['date'] = niftybank_df['date'].dt.strftime('%Y-%m-%d')
        niftybank_df['daily_closing_pct_change'] = niftybank_df['closing_value'].pct_change()
        niftybank_df = niftybank_df.dropna(
            subset='daily_closing_pct_change',
            how='all'
        )
        niftybank_df['daily_log_closing_value'] = cupy.log(1 + niftybank_df['daily_closing_pct_change'])
        niftybank_df['instrument_token'] = instrument_token
        niftybank_df['trading_symbol'] = 'NIFTY BANK'
        print(f'Top 5 rows NIFTY BANK Historical Data (2001-2026):\n{niftybank_df.head()}')
        print(f'Last 5 rows NIFTY BANK Historical Data (2001-2026):\n{niftybank_df.tail()}')
        
        niftybank_df.to_csv(kite_niftybank_daily_historical_data)
        print(f'Loaded {len(niftybank_df)} records in csv file.')
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'Total program runtime: {runtime:.2f} seconds.')
    # Codebase for extracting and transforming (1st round transformation) historical data for GOLD
    elif select_option == 4:
        start_time = time.perf_counter()
        
        # Data extract
        gold_data_dump = yf.download('GC=F', start='2001-01-01', end='2026-01-01', progress=False)

        # Data transformation
        gold_pd = (
            gold_data_dump
            .xs('Close', level='Price', axis=1)
            .rename(columns={'GC=F': 'closing_value'})
            .reset_index()
            .rename(columns={'Date': 'date'})
        )
        gold_df = cudf.from_pandas(gold_pd)
        gold_df.index.name = 'SNo.'
        gold_df['date'] = gold_df['date'].dt.strftime('%Y-%m-%d')
        gold_df['daily_closing_pct_change'] = gold_df['closing_value'].pct_change()
        gold_df = gold_df.dropna(
            subset='daily_closing_pct_change',
            how='all'
        )
        gold_df['daily_log_closing_value'] = cupy.log(1 + gold_df['daily_closing_pct_change'])
        gold_df['exhange_name'] = 'COMEX GOLD FUTURES (CME GROUP)'
        gold_df['trading_symbol'] = 'GC=F'

        print(f'Top 5 rows GOLD PROXY Historical Data (2001-2026):\n{gold_df.head()}')
        print(f'Last 5 rows GOLD PROXY Historical Data (2001-2026):\n{gold_df.tail()}')
        gold_df.to_csv(gold_daily_historical_data)
        print(f'Loaded {len(gold_df)} records in csv file.')
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f'Total program runtime: {runtime:.2f} seconds.')
    else:
        print(f'Invalid selection: {select_option}')

# Main program codebase execution
try:
    main_program()
except Exception as e:
    print(f'Error Encountered: {e}')
    tb = traceback.extract_tb(e.__traceback__)[-1]
    if tb.line == "":
        print(f'Error raised by external package in line number: {tb.lineno}')
    else:
        print(f'Error raised by program in line {tb.line}, number: {tb.lineno}')