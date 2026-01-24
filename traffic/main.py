import os
import asyncio
import random
import time
import enum
from dotenv import load_dotenv
from api_client import ApiClient
from db_logging import log_cpu_mem, get_db_session
from system_monitor import get_system_usage

load_dotenv()

NUM_USERS = int(os.getenv("NUM_USERS", 100))
NUM_COMPANIES = int(os.getenv("NUM_COMPANIES", 10))
REQUEST_TIME_MS = int(os.getenv("TRAFFIC_TIME_REQUEST", 500))
SIMULATION_DURATION_S = int(os.getenv("SIMULATION_DURATION"))

PCT_CAUTIOUS = float(os.getenv("PCT_CAUTIOUS", 0.5))
PCT_ACTIVE = float(os.getenv("PCT_ACTIVE", 0.45))
PCT_BOT = float(os.getenv("PCT_BOT", 0.05))

# Normalizacja gdy nie są w formacie dziesiętnym
if PCT_CAUTIOUS > 1 or PCT_ACTIVE > 1 or PCT_BOT > 1:
    PCT_CAUTIOUS /= 100.0
    PCT_ACTIVE /= 100.0
    PCT_BOT /= 100.0

THRESHOLD_CAUTIOUS = PCT_CAUTIOUS
THRESHOLD_ACTIVE = PCT_CAUTIOUS + PCT_ACTIVE

simulation_active = True


class UserPersona(enum.Enum):
    CAUTIOUS_USER = 0
    ACTIVE_TRADER = 1
    SCRAPER_BOT = 2

class UserStrategy:
    def __init__(self, api: ApiClient, user: dict, companies: list):
        self.api = api
        self.user = user
        self.companies = companies
        self.user_id = user['id']
        self.persona_name = user['persona'].name if 'persona' in user else "UNKNOWN"

    async def run(self):
        print(f"Start symulacji dla użytkownika {self.user_id} (Strategia: {self.__class__.__name__})")
        while simulation_active:
            try:
                await self.execute_action()
                await self.wait()
            except Exception as e:
                print(f"Błąd w strategii {self.__class__.__name__} dla {self.user_id}: {e}")
                await asyncio.sleep(1)

    async def execute_action(self):
        pass

    async def wait(self):
        delay_ms = random.uniform(REQUEST_TIME_MS * 0.8, REQUEST_TIME_MS * 1.2)
        await asyncio.sleep(delay_ms / 1000.0)

    def _get_random_company_id(self) -> int:
        if not self.companies:
            return 1
        return random.choice(self.companies)['id']

class CautiousUserStrategy(UserStrategy):
    async def execute_action(self):
        roll = random.random()
        
        if roll < 0.7:
            browse_action = random.choice([
                lambda: self.api.get_user_stocks(user_id=self.user_id, persona=self.persona_name),
                lambda: self.api.get_all_stock_rates(persona=self.persona_name, user_id=self.user_id),
                lambda: self.api.get_all_companies(persona=self.persona_name, user_id=self.user_id),
                lambda: self.api.get_current_stock_rate(self._get_random_company_id(), persona=self.persona_name, user_id=self.user_id)
            ])
            await browse_action()
        else:
            company_id = self._get_random_company_id()
            amount = random.randint(1, 5)
            
            if random.random() < 0.5:
                price_mod = random.uniform(1.0, 1.05)
                await self.api.simulate_buying(self.user_id, company_id, amount, price_mod, persona=self.persona_name)
            else:
                price_mod = random.uniform(0.95, 1.0)
                await self.api.simulate_selling(self.user_id, company_id, amount, price_mod, persona=self.persona_name)
    
    async def wait(self):
        delay_ms = random.uniform(REQUEST_TIME_MS * 2.0, REQUEST_TIME_MS * 5.0)
        await asyncio.sleep(delay_ms / 1000.0)

class ActiveTraderStrategy(UserStrategy):
    async def execute_action(self):
        roll = random.random()
        company_id = self._get_random_company_id()

        if roll < 0.1:
            await self.api.get_current_stock_rate(
                company_id, 
                persona=self.persona_name, 
                user_id=self.user_id
            )
        else:
            amount = random.randint(5, 15)
            
            if random.random() < 0.6:
                price_mod = random.uniform(1.0, 1.15)
                await self.api.simulate_buying(self.user_id, company_id, amount, price_mod, persona=self.persona_name)
            else:
                price_mod = random.uniform(0.9, 1.0)
                await self.api.simulate_selling(self.user_id, company_id, amount, price_mod, persona=self.persona_name)

    async def wait(self):
        delay_ms = random.uniform(REQUEST_TIME_MS * 0.2, REQUEST_TIME_MS * 0.6)
        await asyncio.sleep(delay_ms / 1000.0)

class ScraperBotStrategy(UserStrategy):
    def __init__(self, api: ApiClient, user: dict, companies: list):
        super().__init__(api, user, companies)
        self.company_ids = [c['id'] for c in companies] if companies else []
        self.current_company_idx = 0

    async def execute_action(self):
        tasks = []
        batch_size = random.randint(50, 100)
        
        if random.random() < 0.9 and self.company_ids:
            for _ in range(batch_size):
                c_id = self.company_ids[self.current_company_idx]
                tasks.append(self.api.get_current_stock_rate(
                    c_id, 
                    persona=self.persona_name,
                    user_id=self.user_id 
                ))
                self.current_company_idx = (self.current_company_idx + 1) % len(self.company_ids)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait(self):
        delay_ms = random.uniform(5000, 15000)
        await asyncio.sleep(delay_ms / 1000.0)
            

STRATEGY_MAP = {
    UserPersona.CAUTIOUS_USER: CautiousUserStrategy,
    UserPersona.ACTIVE_TRADER: ActiveTraderStrategy,
    UserPersona.SCRAPER_BOT: ScraperBotStrategy,
}

def assign_persona_to_user(user: dict) -> UserPersona:
    roll = random.random()
    
    if roll < THRESHOLD_CAUTIOUS:
        return UserPersona.CAUTIOUS_USER
    elif roll < THRESHOLD_ACTIVE:
        return UserPersona.ACTIVE_TRADER
    else:
        return UserPersona.SCRAPER_BOT

async def monitor_system():
    while simulation_active:
        try:
            cpu, mem = get_system_usage()
            log_cpu_mem(cpu, mem)
        except Exception as e:
            print(f"Błąd monitorowania systemu: {e}")
        await asyncio.sleep(5)

async def start_simulation():
    api = ApiClient()
    users = []
    companies = []
    simulation_tasks = []

    await asyncio.sleep(10)

    try:
        session = get_db_session()
        session.close()
    except Exception as e:
        print(f"Nie udało się połączyć z bazą logów: {e}")
        await api.close()
        return

    print(f"Tworzenie {NUM_COMPANIES} firm...")
    company_tasks = [api.create_company() for _ in range(NUM_COMPANIES)]
    created_companies = await asyncio.gather(*company_tasks)
    
    rate_tasks = []
    for company in created_companies:
        if company and 'id' in company:
            companies.append(company)
            rate_tasks.append(api.create_stock_rate(company['id'], 10.0))
    await asyncio.gather(*rate_tasks)
    
    if not companies:
        print("Nie udało się utworzyć żadnej firmy. Przerywam symulację.")
        await api.close()
        return
        
    print(f"Utworzono {len(companies)} firm.")

    print(f"Obliczanie podziału dla {NUM_USERS} użytkowników...")
    
    count_cautious = int(NUM_USERS * PCT_CAUTIOUS)
    count_active = int(NUM_USERS * PCT_ACTIVE)
    
    count_bot = NUM_USERS - count_cautious - count_active
    
    print(f"Planowany podział: Cautious={count_cautious}, Active={count_active}, Bot={count_bot}")

    persona_pool = []
    persona_pool.extend([UserPersona.CAUTIOUS_USER] * count_cautious)
    persona_pool.extend([UserPersona.ACTIVE_TRADER] * count_active)
    persona_pool.extend([UserPersona.SCRAPER_BOT] * count_bot)

    random.seed(42) 
    random.shuffle(persona_pool)

    print(f"Tworzenie {NUM_USERS} użytkowników...")
    user_tasks = [api.create_user() for _ in range(NUM_USERS)]
    created_users = await asyncio.gather(*user_tasks)

    init_tasks = []
    
    for user, persona in zip(created_users, persona_pool):
        if user and 'id' in user:
            user['persona'] = persona 
            
            strategy_class = STRATEGY_MAP[persona]
            strategy_instance = strategy_class(api, user, companies)
            
            simulation_tasks.append(strategy_instance.run())
            
            if persona != UserPersona.SCRAPER_BOT:
                init_tasks.append(api.add_money_to_user(user['id'], 1000000))
                for company in companies:
                    init_tasks.append(api.add_stock(user['id'], company['id'], 100000))

    await asyncio.gather(*init_tasks)
    print(f"Utworzono {len(simulation_tasks)} użytkowników i zainicjalizowano ich konta.")

    print("Rozpoczynanie symulacji rynkowej...")
    
    try:
        await asyncio.wait_for(asyncio.gather(*simulation_tasks), timeout=SIMULATION_DURATION_S)
    except asyncio.TimeoutError:
        print(f"Zakończono symulację po {SIMULATION_DURATION_S} sekundach.")
    finally:
        global simulation_active
        simulation_active = False 
        await api.close()
        print("Klient API zamknięty.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    
    monitor_task = loop.create_task(monitor_system())
    
    try:
        loop.run_until_complete(start_simulation())
    except KeyboardInterrupt:
        simulation_active = False
    finally:
        monitor_task.cancel()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        print("Symulacja zakończona.")