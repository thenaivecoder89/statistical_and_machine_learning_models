from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Request

# Initialize environment
load_dotenv()
kite_api = os.getenv('kite_connect_api').strip()
kite_secret = os.getenv('kite_connect_secret').strip()

kc = KiteConnect(api_key=kite_api)
app = FastAPI()

@app.get('/kite/callback')
def connect_kite(requests: Request):
    try:
        request_token = requests.query_params.get('request_token')
        if not request_token:
            raise ValueError('Missing request_token in query')
        
        sess = kc.generate_session(request_token=request_token, api_secret=kite_secret)
        kc.set_access_token(sess['access_token'])

        # write access token to file to persist for the day
        with open('kite_access_token.txt', 'w') as f_kc_access_token:
            f_kc_access_token.write(sess['access_token'])
        
        return {'status': 'ok'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/healthz')
def get_health():
    return {'status': 'ok'}

@app.get('/login_url')
def get_login_url():
    return {'login_url': kc.login_url()}