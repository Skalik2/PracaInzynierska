import express from "express";
import dotenv from "dotenv";
import "./database/dataSource";
import { AppDataSource, initializeDatabase } from "./database/dataSource";
import { globalErrorHandler } from "./middleware/globalErrorHandler";
import { APPROUTER } from "./appRouters";
import { createStockLog } from "./utils/logger/createlog";
import { initializeLogDatabase, LogAppDataSource } from "./database/logDB/logDataSource";
import { securityMiddleware } from "./middleware/securityMiddleware";

const app = express();
app.use(express.json());
dotenv.config({ path: `${process.cwd()}/./.env` });
const PORT = process.env.APP_PORT || 3000;

//
app.use(securityMiddleware);
APPROUTER.forEach(({ path, router }) => app.use(`/api/${path}`, router));

app.use(globalErrorHandler);

async function startServer() {
  try {
    await initializeLogDatabase();
    console.log("Log database initialized");

    await initializeDatabase();
    console.log("Main database initialized");

    app.listen(PORT, () => {
      console.log("Server is running on http://localhost:" + PORT);
    });

    setInterval(async () => {
      await createStockLog();
    }, 5000);
  } catch (error) {
    console.error("Error during initialization:", error);
  }
}

startServer();
