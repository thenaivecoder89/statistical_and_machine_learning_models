from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os, pandas as pd
import requests
import time
import math


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
    run_requirement = int(input('Select program to run 1-Kite_Data, 2-Specific_MF_Data or 3-Full_MF_Data: '))
    if run_requirement == 1:
        amfi_data_dump_full = []
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
    elif run_requirement == 2:
    # Source data from AMFI Community Mirror
        # Pull relevant scheme code from API
        try:
            search_url = 'https://api.mfapi.in/mf' # community maintained mirror of official Association of Mutual Funds of India (AMFI) data
            resp = requests.get(search_url, timeout=20)
            try:
                if resp.status_code == 200:
                    out = resp.json()
                    fund_name = input('Enter fund name: ')
                    for funds in out:
                        if fund_name in funds['schemeName']:
                            scheme_code = funds['schemeCode']
                        else:
                            pass
                else:
                    print(f'{resp.status_code} Error')
            except requests.exceptions.Timeout:
                print('Error, Session Timeout')
            
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
            print(f'Error Encountered While Executing Option 2: {e}')
    elif run_requirement == 3:
        start_time = time.time()
    # Source entire repo of AMFI data
        scheme_code_all = []
        nav_data_url_full = []
        amfi_data_dump_full = []
        fund_name = []
        counter = 1
        search_url = 'https://api.mfapi.in/mf'
        response = requests.get(search_url)
        out = response.json()
        print('Commencing URL Generation..')
        scheme_code_all = [codes['schemeCode'] for codes in out]
        scheme_code_all = list(set(scheme_code_all))
        nav_data_url_full = [
            f'https://api.mfapi.in/mf/{scheme_code}' for scheme_code in scheme_code_all
        ]
        total_records = len(nav_data_url_full)
        print(f'URLs fetched: {total_records}')
        print(f'Commencing data download for {total_records}...')
        batch_size = int(input('Enter batch size (in multiples of 10): '))
        total_batch = math.ceil(total_records / batch_size)
        print(f'Total batches: {total_batch}')
        batch = 1
        while batch <= total_batch:
            cont = input(f'Processing batch {batch} of {total_batch}. Continue? (Y/N): ')
            if cont == 'Y':
                mode = input('Would you like to manually select batch start and end? (Y/N):')
                if mode == 'Y':
                    batch_start = int(input('Enter batch start number: '))
                    batch_end = int(input('Enter batch end number: '))
                else:
                    batch_start = (batch - 1) * batch_size
                    batch_end = min((batch * batch_size), total_records)
                
                current_batch_urls = nav_data_url_full[batch_start:batch_end]
                loop_range = len(current_batch_urls)
                for i in range(loop_range):
                    print(f'Counter: {i + 1} of {loop_range}')
                    url = current_batch_urls[i]
                    response_full = requests.get(url, timeout=20)
                    if response_full.status_code == 200:
                        try:
                            output_full = response_full.json()
                            fund_name = output_full['meta'].get('fund_house')
                            fund_scheme_code = output_full['meta'].get('scheme_code')
                            fund_scheme_name = output_full['meta'].get('scheme_name')
                            fund_trading_symbol_growth = output_full['meta'].get('isin_growth')
                            fund_trading_symbol_reinvestment = output_full['meta'].get('isin_div_reinvestment')
                            df_full_data = pd.DataFrame(output_full['data'])
                            df_full_data['fund_name'] = fund_name
                            df_full_data['scheme_code'] = fund_scheme_code
                            df_full_data['scheme_name'] = fund_scheme_name
                            df_full_data['trading_symbol_growth'] = fund_trading_symbol_growth
                            df_full_data['trading_symbol_reinvestment'] = fund_trading_symbol_reinvestment
                            amfi_data_dump_full.append(df_full_data)
                            # print(f'Output:\n{amfi_data_dump_full}')
                        except ValueError:
                            print(f'Empty or invalid JSON for URL: {url}')
                            df_data = pd.concat(amfi_data_dump_full)
                            df_data.index.name = 'SNo.'
                            df_data.to_csv(f'{mf_hist_nav_data}/A.AMFI_FULL_DATA_DUMP.csv')
                            continue
                        except requests.exceptions.Timeout:
                            print(f'Error, Session Timeout')
                            df_data = pd.concat(amfi_data_dump_full)
                            df_data.index.name = 'SNo.'
                            df_data.to_csv(f'{mf_hist_nav_data}/A.AMFI_FULL_DATA_DUMP.csv')
                            continue
                        except Exception as e:
                            print(f'Error Encontered: {e}')
                            df_data = pd.concat(amfi_data_dump_full)
                            df_data.index.name = 'SNo.'
                            df_data.to_csv(f'{mf_hist_nav_data}/A.AMFI_FULL_DATA_DUMP.csv')
                            continue
                    else:
                        print(f'Request failed for URL: {url}, status code: {response_full.status_code}')
                        continue
                    time.sleep(1)
            else:
                print(f'Discontinuing processing.')
                break
            print(f'Completed batch {batch} of {total_batch}.')
            batch += 1
        print('Data download completed.')
        df_data_extract = pd.concat(amfi_data_dump_full, ignore_index=True)
        df_data_extract.index.name = 'SNo.'
        df_data_extract.to_csv(f'{mf_hist_nav_data}/A.AMFI_FULL_DATA_DUMP.csv')
        end_time = time.time()
        total_pgm_time = end_time - start_time
        print(f'Total time taken for program execution: {total_pgm_time:.2f}')
    else:
        print('Invalid choice. Please re-run the program')
except Exception as e:
    print(f'Error Encountered: {e}')
    df_data = pd.concat(amfi_data_dump_full)
    df_data.index.name = 'SNo.'
    df_data.to_csv(f'{mf_hist_nav_data}/A.AMFI_FULL_DATA_DUMP.csv')
    print(f'Pulled data stored as CSV. Program Terminated.')