import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect
import pandas as pd
import sys, cudf, time, requests, traceback, io

class AmfiDataPipeline:

    def __init__(self, url: str, timeout = 20):
        self.url = url
        self.timeout = timeout

    def data_load(self, amfi_filename: str, api: str, kite_filename: str):
        start_time = time.perf_counter() # Record code block start time
        kc = KiteConnect(api)
        resp = requests.get(self.url, timeout=self.timeout)
        resp.raise_for_status() # Raises built-in HTTP error if status code != 200
        data = io.StringIO(resp.text) # To allow CuDF to read the data and enable GPU transformations
        cudf_amfi_data = cudf.read_json(data)
        cudf_amfi_data.to_csv(amfi_filename)
        df_kite_mf_data = pd.DataFrame(kc.mf_instruments()) # To handle datetime
        df_kite_mf_data['last_price_date'] = pd.to_datetime(df_kite_mf_data['last_price_date'])
        cudf_kite_mf_data = cudf.DataFrame(df_kite_mf_data)
        cudf_kite_mf_data.to_csv(kite_filename)
        end_time = time.perf_counter() # Record code block end time
        runtime = end_time - start_time
        return cudf_amfi_data, cudf_kite_mf_data, runtime
    
    def data_transformation(self, cudf_amfi_data: cudf.DataFrame, cudf_kite_data: cudf.DataFrame, filename: str):
        start_time = time.perf_counter() # Record code block start time
        cudf_amfi_data = cudf_amfi_data.dropna(
            subset=['isinGrowth', 'isinDivReinvestment'],
            how='all'
        )
        print(f'Remaining data after 1st transformation (drop NULL values in isinGrowth and isinDivReinvestment): {len(cudf_amfi_data)}')
        cudf_amfi_data['URLs'] = self.url + '/' + cudf_amfi_data['schemeCode'].astype('str')
        cudf_amfi_data = cudf.merge(
            cudf_amfi_data,
            cudf_kite_data,
            left_on='isinGrowth',
            right_on='tradingsymbol',
            how='left'
        )
        print(f'Remaining data after 2nd transformation (merge AMFI and KITE data): {len(cudf_amfi_data)}')
        cudf_transformed_data = cudf_amfi_data[
            [
                'tradingsymbol', 
                'isinGrowth',
                'isinDivReinvestment',
                'amc',
                'name',
                'schemeCode', 
                'schemeName', 
                'dividend_type',
                'scheme_type',
                'plan',
                'URLs'
            ]
        ]
        cudf_transformed_data = cudf_transformed_data.dropna(
            subset=['amc', 'name'],
            how='all'
        )
        print(f'Remaining data after 3rd transformation (drop NULL values in amc and name): {len(cudf_transformed_data)}')
        cudf_transformed_data.to_csv(filename)
        end_time = time.perf_counter()  # Record code block end time
        runtime = end_time - start_time
        return cudf_transformed_data, runtime

    def nav_data_extract(self, cudf_amfi_trx_data: cudf.DataFrame, filename: str):
        start_time = time.perf_counter() # Record code block start time
        print(f'Total number of records for NAV data extract: {len(cudf_amfi_trx_data)}')
        urls_list = cudf_amfi_trx_data['URLs'].to_arrow().to_pylist()
        master_data = []
        counter = 1
        for item in range(len(urls_list)):
            try:
                print(f'Pulling data from URL: {urls_list[item]}')
                resp = requests.get(urls_list[item], timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                schemeCode_NAV = data['meta']['scheme_code']
                # Flattening each row, attaching scheme code and appending to master list
                for nav_row in data['data']:
                    nav_row['schemeCode_NAV'] = schemeCode_NAV
                    master_data.append(nav_row)

                print(f'Processed {counter} of {len(urls_list)}.')
                counter += 1
            except Exception as e:
                print(f'Error encountered when processing record number {counter}. Proceeding to next counter.')
                tb = traceback.extract_tb(e.__traceback__)[-1]
                print(f'Error: {e} in line {tb.line} number: {tb.lineno}')
                counter += 1
                continue
        # df_nav_amfi_data = pd.DataFrame(master_data['data'])
        # df_nav_amfi_data['schemeCode_NAV'] = master_data['schemeCode_NAV']
        cudf_nav_amfi_data = cudf.DataFrame(master_data)
        print(f'Total records with NAV data: {len(cudf_nav_amfi_data)}')
        cudf_nav_amfi_data = cudf.merge(
            cudf_nav_amfi_data,
            cudf_amfi_trx_data,
            left_on='schemeCode_NAV',
            right_on='schemeCode',
            how='left'
        )
        cudf_nav_amfi_data = cudf_nav_amfi_data[
            [
                'schemeCode_NAV',
                'tradingsymbol',
                'isinGrowth',
                'isinDivReinvestment',
                'amc',
                'name',
                'schemeName',
                'dividend_type',
                'scheme_type',
                'plan',
                'date',
                'nav'
            ]
        ]
        print(f'Total records in final dataset - including NAV data: {len(cudf_nav_amfi_data)}')
        cudf_nav_amfi_data.to_csv(filename)
        end_time = time.perf_counter() # Record code block end time
        runtime = end_time - start_time
        return cudf_nav_amfi_data, runtime

# Main program block
def main_program():
    # Initialize environment variables
    load_dotenv()
    amfi_url = os.getenv('amfi_community_data')
    amfi_base_data = os.getenv('amfi_base_data')
    kite_base_data = os.getenv('kite_base_data')
    amfi_trx_data = os.getenv('amfi_trx_data')
    kite_connect_api = os.getenv('kite_connect_api')
    amfi_nav_data = os.getenv('amfi_nav_data')

    # Initialize class
    amfi_pipeline = AmfiDataPipeline(url=amfi_url, timeout=30)
    
    run_option = int(input("""Available options: 
                           1-Data_Load,
                           2-Data_Transformation,
                           3-NAV_Data_Extract
                           Select option: """))
    if run_option == 1: # Data Load
        # Call data load function
        dl_output, dl_kite_output, dl_run = amfi_pipeline.data_load(
            amfi_filename=amfi_base_data, 
            api=kite_connect_api, 
            kite_filename=kite_base_data
        )
        print(f'Loaded {len(dl_output)} records from AMFI. Top 10 records:\n{dl_output.head(10)}')
        print(f'Loaded {len(dl_kite_output)} records from KITE. Top 10 records:\n{dl_kite_output.head(10)}')
        print(f'Data load execution time: {dl_run:.2f} seconds.')
    elif run_option == 2: # Data Transformation
        dl_amfi_base = cudf.read_csv(amfi_base_data) # Load base data - AMFI
        dl_kite_base = cudf.read_csv(kite_base_data) # Load base data - KITE
        # Call data transformation function
        dt_output, dt_run = amfi_pipeline.data_transformation(
            cudf_amfi_data=dl_amfi_base,
            cudf_kite_data=dl_kite_base,
            filename=amfi_trx_data
        )
        print(f'Records remaining after transformation: {len(dt_output) / len(dl_amfi_base) * 100:.2f}%. Top 10 records:\n{dt_output.head(10)}')
        print(f'Data transformation execution time: {dt_run:.2f} seconds.')
    elif run_option == 3: # NAV Data Extract
        dt_output = cudf.read_csv(amfi_trx_data) # Load transformed data
        # Call nav data extract function
        df_nav_extract, df_nav_run = amfi_pipeline.nav_data_extract(
            cudf_amfi_trx_data=dt_output,
            filename=amfi_nav_data
        )
        print(f'NAV dataset:\n{df_nav_extract.head(10)}')
        print(f'Loaded {len(df_nav_extract)} records into CSV.')
        print(f'Code execution time: {df_nav_run/60:.2f} minutes.')
    else:
        print('Invalid selection')

# Program execution and exception handling block
try:
    main_program()
except Exception as e:
    print(f'Error encountered: {e}')
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(f'Error in line number: {tb.lineno} and code: {tb.line}')