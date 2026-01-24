import aiohttp
import uuid
import time
import string
import random
from datetime import datetime, timedelta
from db_logging import log_api_call

API_URL = "http://market_service:3000/api"

def _generate_string(length: int) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class ApiClient:
    def __init__(self):
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=0) 
        self.session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        await self.session.close()

    async def _request(self, method: str, endpoint: str, data: dict = None, persona: str = None, user_id: int = None) -> dict:
        request_id = str(uuid.uuid4())
        headers = {'X-Request-Id': request_id}
        
        if persona:
            headers['X-User-Persona'] = persona

        if user_id is not None:
            headers['X-User-Id'] = str(user_id)

        start_time = time.monotonic()
        
        url = f"{API_URL}{endpoint}"
        
        try:
            async with self.session.request(method, url, json=data, headers=headers) as response:
                end_time = time.monotonic()
                duration_ms = int((end_time - start_time) * 1000)
                
                log_api_call(request_id, duration_ms)
                
                response.raise_for_status()
                return await response.json()
        
        except aiohttp.ClientResponseError as e:
            if method != "GET":
                print(f"Błąd API {method} {endpoint}: {e.status} {e.message}")
            return {"error": e.message}
        except Exception as e:
            print(f"Błąd zapytania {method} {endpoint}: {e}")
            return {"error": str(e)}

    async def create_user(self, persona: str = None) -> dict:
        text = _generate_string(6)
        payload = {
            "name": text,
            "surname": text,
            "username": text,
            "password": text,
            "email": f"{text}@example.com"
        }
        data = await self._request("POST", "/user/create", payload, persona=persona)
        return data.get("result")

    async def add_money_to_user(self, user_id: int, money: float):
        await self._request("POST", "/user/money", {"userId": user_id, "money": money})

    async def create_company(self) -> dict:
        payload = {"name": _generate_string(6)}
        data = await self._request("POST", "/company/create", payload)
        return data.get("result") 

    async def add_stock(self, user_id: int, company_id: int, amount: int):
        await self._request("POST", "/stock/create", {"companyId": company_id, "userId": user_id, "amount": amount})

    async def create_stock_rate(self, company_id: int, rate: float):
        await self._request("POST", "/stockrate/create", {"companyId": company_id, "rate": rate})

    async def get_current_stock_rate(self, company_id: int, persona: str = None, user_id: int = None) -> float | None:
        try:
            url = f"{API_URL}/stockrate/company/{company_id}"
            request_id = str(uuid.uuid4())
            headers = {'X-Request-Id': request_id}
            
            if persona:
                headers['X-User-Persona'] = persona

            if user_id is not None:
                headers['X-User-Id'] = str(user_id)

            start_time = time.monotonic()

            async with self.session.get(url, headers=headers) as response:
                end_time = time.monotonic()
                duration_ms = int((end_time - start_time) * 1000)
                log_api_call(request_id, duration_ms)
                
                text_data = await response.text()
                
                if not text_data or text_data == "null":
                    return None
                
                clean_text = text_data.strip('"').strip("'")
                
                try:
                    rate_data = float(clean_text)
                    return rate_data
                except ValueError:
                    try:
                        json_data = aiohttp.helpers.json_loads(text_data)
                        if isinstance(json_data, dict) and "message" in json_data:
                            return None
                    except Exception:
                        pass 
                    
                return None
        except Exception as e:
            return None

    async def simulate_buying(self, user_id: int, company_id: int, amount: int, price_modifier: float, persona: str = None):
        current_rate = await self.get_current_stock_rate(company_id, persona=persona, user_id=user_id)
        if current_rate is None or current_rate <= 0:
            return

        max_price = round(current_rate * price_modifier, 2)
        date_limit = (datetime.now() + timedelta(minutes=6)).isoformat()
        
        payload = {
            "userId": user_id,
            "companyId": company_id,
            "max_price": max_price,
            "amount": amount,
            "date_limit": date_limit
        }
        await self._request("POST", "/buyoffer/create", payload, persona=persona, user_id=user_id)

    async def simulate_selling(self, user_id: int, company_id: int, amount: int, price_modifier: float, persona: str = None):
        current_rate = await self.get_current_stock_rate(company_id, persona=persona, user_id=user_id)
        if current_rate is None or current_rate <= 0:
            return

        min_price = round(current_rate * price_modifier, 2)
        date_limit = (datetime.now() + timedelta(minutes=6)).isoformat()
        
        payload = {
            "userId": user_id,
            "companyId": company_id,
            "min_price": min_price,
            "amount": amount,
            "date_limit": date_limit
        }
        await self._request("POST", "/selloffer/create", payload, persona=persona, user_id=user_id)


    async def get_user_stocks(self, user_id: int, persona: str = None):
        return await self._request("GET", f"/stock/user/{user_id}", persona=persona, user_id=user_id)

    async def get_all_stock_rates(self, user_id: int, persona: str = None):
        return await self._request("GET", "/stockrate/all", persona=persona, user_id=user_id)

    async def get_all_companies(self, user_id: int, persona: str = None):
        return await self._request("GET", "/company/allcompany", persona=persona, user_id=user_id)
    
    async def get_all_transactions(self, user_id: int, persona: str = None):
        return await self._request("GET", "/transaction/all", persona=persona, user_id=user_id)

    async def get_all_users(self, user_id: int, persona: str = None):
        return await self._request("GET", "/user/allusers", persona=persona, user_id=user_id)