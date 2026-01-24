import dotenv from "dotenv";
import { DataSource } from "typeorm";
import { createDatabase, dropDatabase } from "typeorm-extension";
import { MarketCpu } from "./entities/MarketCpuEnitities";
import { MarketLog } from "./entities/MarketLogEntities";
import { TradeCpu } from "./entities/TradeCpuEnitities";
import { TradeLog } from "./entities/TradeLogEntities";
import { TrafficCpu } from "./entities/TrafficCpuEntities";
import { TrafficLog } from "./entities/TrafficLogEntities";
import { BlockedUserLog } from "./entities/BlockedUserLogEntities";

dotenv.config({ path: `${process.cwd()}/./.env` });

const entities = [MarketCpu, MarketLog, TradeCpu, TradeLog, TrafficCpu, TrafficLog, BlockedUserLog];

export const LogAppDataSource = new DataSource({
  type: "postgres",
  host: process.env.DB_HOST2,
  port: Number(process.env.DB_PORT2),
  username: process.env.DB_USERNAME,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  entities,
  synchronize: true,
  logging: false,
});

export async function initializeLogDatabase() {
  await dropDatabase({ options: LogAppDataSource.options });
  await createDatabase({ options: LogAppDataSource.options });
  await LogAppDataSource.initialize();
}
