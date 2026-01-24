import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError

DB_HOST = os.getenv("DB_HOST2", "logdatabase")
DB_PORT = os.getenv("DB_PORT2", 5432)
DB_USER = os.getenv("DB_USERNAME", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "Password")
DB_NAME = os.getenv("DB_NAME", "gielda")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Błąd połączenia z bazą logów: {e}")
    exit(1)

class Base(DeclarativeBase):
    pass

class TrafficLog(Base):
    __tablename__ = 'traffic_log'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    requestId = Column(String)
    apiTime = Column(Integer)

class TrafficCpu(Base):
    __tablename__ = 'traffic_cpu'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    cpuUsage = Column(Numeric(5, 2))
    memoryUsage = Column(Numeric(5, 2))

class BotStats(Base):
    __tablename__ = 'bot_stats'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    totalUsers = Column(Integer)
    blockedUsers = Column(Integer)
    activeUsers = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db_session():
    return SessionLocal()

def log_api_call(request_id: str, api_time_ms: int):
    session = get_db_session()
    try:
        log_entry = TrafficLog(requestId=request_id, apiTime=api_time_ms, timestamp=datetime.now())
        session.add(log_entry)
        session.commit()
    except SQLAlchemyError as e:
        print(f"Błąd zapisu logu API: {e}")
        session.rollback()
    finally:
        session.close()

def log_cpu_mem(cpu_usage: float, mem_usage: float):
    session = get_db_session()
    try:
        log_entry = TrafficCpu(cpuUsage=cpu_usage, memoryUsage=mem_usage, timestamp=datetime.now())
        session.add(log_entry)
        session.commit()
    except SQLAlchemyError as e:
        print(f"Błąd zapisu logu CPU: {e}")
        session.rollback()
    finally:
        session.close()

def log_bot_stats(total: int, blocked: int, active: int):
    session = get_db_session()
    try:
        entry = BotStats(
            totalUsers=total,
            blockedUsers=blocked,
            activeUsers=active,
            timestamp=datetime.now()
        )
        session.add(entry)
        session.commit()
    except SQLAlchemyError as e:
        print(f"Błąd zapisu statystyk botów: {e}")
        session.rollback()
    finally:
        session.close()