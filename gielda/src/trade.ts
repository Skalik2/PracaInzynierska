import dotenv from "dotenv";
import { AppDataSource } from "./database/dataSource";
import { trade } from "./services/tradeService";
import { createTradeCpuLog } from "./utils/logger/createlog";
import { LogAppDataSource } from "./database/logDB/logDataSource";

dotenv.config({ path: `${process.cwd()}/./.env` });

AppDataSource.initialize()
  .then(async () => {
    await LogAppDataSource.initialize();
    console.log("Log database initialized");
    trade();
    setInterval(async () => {
      await createTradeCpuLog(Number(process.env.TRADE_ID));
    }, 5000);
  })
  .catch((error) => console.log(error));
