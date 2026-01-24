import { MarketCpu } from "../entities/MarketCpuEnitities";
import { ApiMethod, MarketLog } from "../entities/MarketLogEntities";
import { TradeCpu } from "../entities/TradeCpuEnitities";
import { TradeLog } from "../entities/TradeLogEntities";
import { TrafficCpu } from "../entities/TrafficCpuEntities";
import { TrafficLog } from "../entities/TrafficLogEntities";

export interface CpuLog {
  cpuUsage: number;
  memoryUsage: number;
}
const addMarketCpuLog = async (timestamp: Date, data: CpuLog) => {
  await MarketCpu.save({
    timestamp,
    cpuUsage: data.cpuUsage,
    memoryUsage: data.memoryUsage,
  });
};

const addTrafficCpu = async (timestamp: Date, data: CpuLog) => {
  await TrafficCpu.save({
    timestamp,
    cpuUsage: data.cpuUsage,
    memoryUsage: data.memoryUsage,
  });
};

export interface TradeCpuLog extends CpuLog {
  replicaId: number;
}
const addTradeCpuLog = async (timestamp: Date, data: TradeCpuLog) => {
  await TradeCpu.save({
    timestamp,
    cpuUsage: data.cpuUsage,
    memoryUsage: data.memoryUsage,
    replicaId: data.replicaId,
  });
};

export interface APILog {
  apiMethod: ApiMethod;
  applicationTime: number;
  databaseTime: number;
  endpointUrl: string;
  requestId: string;
  userId?: number;
  userPersona?: string;
}
const addMarketLog = async (timestamp: Date, data: APILog) => {
  await MarketLog.save({
    timestamp,
    apiMethod: data.apiMethod,
    applicationTime: data.applicationTime,
    databaseTime: data.databaseTime,
    endpointUrl: data.endpointUrl,
    requestId: data.requestId,
    userId: data.userId,
    userPersona: data.userPersona,
  });
};

export interface TradeLogData {
  applicationTime: number;
  databaseTime: number;
  numberOfSellOffers: number;
  numberOfBuyOffers: number;
}
const addTradeLog = async (timestamp: Date, data: TradeLogData) => {
  await TradeLog.save({
    timestamp,
    applicationTime: data.applicationTime,
    databaseTime: data.databaseTime,
    numberOfSellOffers: data.numberOfSellOffers,
    numberOfBuyOffers: data.numberOfBuyOffers,
  });
};

export interface TrafficLogData {
  requestId: string;
  apiTime: number;
}
const addTrafficLog = async (timestamp: Date, data: TrafficLogData) => {
  await TrafficLog.save({
    timestamp,
    requestId: data.requestId,
    apiTime: data.apiTime,
  });
};

export type LogType = "marketCpu" | "trafficCpu" | "tradeCpu" | "marketLog" | "tradeLog" | "trafficLog";

export type LogData = CpuLog | TradeCpuLog | APILog | TradeLogData | TrafficLogData;

const logFunctions: Record<LogType, (timestamp: Date, data: any) => Promise<void>> = {
  marketCpu: addMarketCpuLog,
  trafficCpu: addTrafficCpu,
  tradeCpu: addTradeCpuLog,
  marketLog: addMarketLog,
  tradeLog: addTradeLog,
  trafficLog: addTrafficLog,
};

export const addLogToDB = async (logType: LogType, timestamp: Date, data: LogData) => {
  const logFunction = logFunctions[logType];

  if (!logFunction) {
    throw new Error(`Log function for type ${logType} not found`);
  }

  await logFunction(timestamp, data);
};
