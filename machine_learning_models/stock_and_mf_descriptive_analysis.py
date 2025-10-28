import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Initialize environment
load_dotenv()
mf_hist_nav_data = os.getenv('mf_hist_nav_data')
motilal_hist_nav_data = os.getenv('motilal_hist_nav_data')
aditya_birla_sun_life_hist_nav_data = os.getenv('aditya_birla_sun_life_hist_nav_data')
pd.set_option('display.max_rows', None)

try:
    # Load data
    df_motilal = pd.read_csv(motilal_hist_nav_data)
    df_aditya_birla_sun_life = pd.read_csv(aditya_birla_sun_life_hist_nav_data)
    print(f'Descriptive Statistics on Motilal Oswal Midcap Fund:\n{df_motilal.describe()}\n')
    print(f'Descriptive Statistics on Aditya Aditya Birla Sun Life Liquid Fund:\n{df_aditya_birla_sun_life.describe()}')
except Exception as e:
    print(f'Error encountered: {e}')