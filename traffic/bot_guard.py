from aiohttp import web
import xgboost as xgb
import numpy as np
import joblib
import asyncio
import pandas as pd
from dotenv import load_dotenv

from system_monitor import get_system_usage
from db_logging import log_cpu_mem, log_bot_stats

load_dotenv()

model = xgb.Booster()
model.load_model("bot_xgboost_model_H1U200-45-45-10.json")

encoders_data = joblib.load("bot_xgboost_encoders_H1U200-45-45-10.pkl")
url_counts = encoders_data['url_counts']
ohe_encoder = encoders_data['ohe_encoder']

BOT_THRESHOLD = 0.90

stats_lock = asyncio.Lock()
seen_users = set()
blocked_users = set()

system_state = {
    'MARKET': {'cpu': 10.0, 'mem': 30.0},
    'TRADE':  {'cpu': 10.0, 'mem': 30.0}
}

async def predict(request):
    data = await request.json()
    
    user_id = data.get('userId')
    raw_url = data.get('endpointUrl', '')
    raw_method = data.get('apiMethod', '')

    if raw_url in url_counts.index:
        url_enc = url_counts[raw_url]
    else:
        url_enc = 0

    try:
        input_df = pd.DataFrame([[raw_method]], columns=['apiMethod'])
        method_enc_vector = ohe_encoder.transform(input_df)[0]
    except Exception:
        method_enc_vector = np.zeros(len(ohe_encoder.categories_[0]))

    incoming_cpu = data.get('cpuUsage', 0)
    incoming_mem = data.get('memoryUsage', 0)
    service_type = data.get('serviceType', 'MARKET').upper()

    if incoming_cpu <= 0:
        try:
            real_cpu, _ = get_system_usage()
            incoming_cpu = real_cpu
        except:
            incoming_cpu = 10.0

    if service_type in system_state:
        system_state[service_type]['cpu'] = incoming_cpu
        system_state[service_type]['mem'] = incoming_mem

    cpu_market = system_state['MARKET']['cpu']
    cpu_trade  = system_state['TRADE']['cpu']
    mem_market = system_state['MARKET']['mem']
    mem_trade  = system_state['TRADE']['mem']

    base_features = [
        data.get('apiTime', 0),
        data.get('applicationTime', 0),
        data.get('databaseTime', 0),
        cpu_market,
        cpu_trade,
        mem_trade,
        mem_market,
        url_enc
    ]

    final_features = np.concatenate([base_features, method_enc_vector])
    features_matrix = np.array([final_features])

    base_names = [
        'apiTime', 'applicationTime', 'databaseTime', 
        'cpuUsage_market', 'cpuUsage_trade', 
        'memoryUsage_trade', 'memoryUsage_market', 
        'endpointUrl'
    ]
    method_names = list(ohe_encoder.get_feature_names_out(['apiMethod']))
    
    full_feature_names = base_names + method_names

    dmatrix = xgb.DMatrix(features_matrix, feature_names=full_feature_names)
    prediction = model.predict(dmatrix)[0]
    
    is_bot = prediction > BOT_THRESHOLD
    
    if user_id:
        async with stats_lock:
            if user_id not in seen_users:
                seen_users.add(user_id)
            
            is_newly_blocked = False
            if is_bot and user_id not in blocked_users:
                blocked_users.add(user_id)
                is_newly_blocked = True
            
            if is_newly_blocked:
                total = len(seen_users)
                blocked = len(blocked_users)
                active = total - blocked
                try:
                    log_bot_stats(total, blocked, active)
                    print(f"[DB SAVED] New block! User: {user_id}. Stats: {blocked}/{total}")
                except Exception as e:
                    print(f"Failed to log bot stats: {e}")
    
    if is_bot:
        print(f"[BLOCK] User: {user_id} | Score: {prediction:.4f}")

    return web.json_response({'is_bot': bool(is_bot), 'score': float(prediction)})

async def monitor_system(app):
    try:
        while True:
            try:
                cpu, mem = get_system_usage()
                log_cpu_mem(cpu, mem)
            except Exception as e:
                print(f"Błąd monitorowania systemu: {e}")

            await asyncio.sleep(5)
            
    except asyncio.CancelledError:
        print("Zatrzymano monitorowanie.")

async def start_background_tasks(app):
    app['monitor_task'] = asyncio.create_task(monitor_system(app))

async def cleanup_background_tasks(app):
    app['monitor_task'].cancel()
    await app['monitor_task']

app = web.Application()
app.add_routes([web.post('/predict', predict)])

app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)

if __name__ == '__main__':
    web.run_app(app, port=5000)