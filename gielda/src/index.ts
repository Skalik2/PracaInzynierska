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

// TO ZMIANA
app.use(securityMiddleware);
APPROUTER.forEach(({ path, router }) => app.use(`/api/${path}`, router));

app.use(globalErrorHandler);

async function startServer() {
  try {
    // Inicjalizacja bazy logów po zakończeniu inicjalizacji głównej bazy
    await initializeLogDatabase();
    console.log("Log database initialized");

    // Inicjalizacja głównej bazy danych
    await initializeDatabase();
    console.log("Main database initialized");

    // Gdy obie bazy zostaną zainicjalizowane, uruchom serwer
    app.listen(PORT, () => {
      console.log("Server is running on http://localhost:" + PORT);
    });

    // Co 5 sekund twórz logi
    setInterval(async () => {
      await createStockLog(); // Zakładam, że createStockLog zapisuje logi do bazy logów
    }, 5000);
  } catch (error) {
    console.error("Error during initialization:", error);
  }
}

startServer();
