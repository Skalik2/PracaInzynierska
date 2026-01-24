import si from "systeminformation";
import { log } from "./logger";
import { APILog, CpuLog, LogType, TradeCpuLog, TradeLogData } from "../../database/logDB/services/addLog";

let lastCpuUsage = 0.0;

export const getLastCpuUsage = () => lastCpuUsage;

export const createLog = async (info: APILog | TradeLogData, logType: LogType) => {
  log("info", info, logType);
};

const cpuMemoryUse = async () => {
  const cpuLoad = await si.currentLoad();
  const memory = await si.mem();

  const cpuUsage = parseFloat(cpuLoad.currentLoad.toFixed(2));
  const memoryUsage = parseFloat(((memory.active / memory.total) * 100).toFixed(2));

  lastCpuUsage = cpuUsage;
  const message: CpuLog = {
    cpuUsage,
    memoryUsage,
  };

  return message;
};

export const createStockLog = async () => {
  const message = await cpuMemoryUse();
  log("info", message, "marketCpu");
};

export const createTradeCpuLog = async (replicaId: number) => {
  const cpuMemory = await cpuMemoryUse();
  const message: TradeCpuLog = { ...cpuMemory, replicaId };

  log("info", message, "tradeCpu");
};
