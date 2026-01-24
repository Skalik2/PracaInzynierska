import { createLogger, format, transports } from "winston";
import "winston-daily-rotate-file";
import { existsSync, mkdirSync } from "fs";
import { addLogToDB, LogData, LogType } from "../../database/logDB/services/addLog";

const logDir = "logs";
if (!existsSync(logDir)) {
  mkdirSync(logDir);
}

const logger = createLogger({
  format: format.combine(format.printf(({ level, message, timestamp }) => `${timestamp} ${level}: ${message}`)),
  transports: [
    new transports.Console(),
    new transports.DailyRotateFile({
      filename: `${logDir}/application-%DATE%.log`,
      datePattern: "YYYY-MM-DD",
      zippedArchive: true,
      maxSize: "20m",
      maxFiles: "14d",
    }),
  ],
});

export const log = async (level: string, message: LogData, logType: LogType) => {
  const currentTimestamp = new Date();

  const messageString = JSON.stringify(message);
  logger.log({
    level,
    message: messageString,
    timestamp: currentTimestamp.toISOString(),
  });

  await addLogToDB(logType, currentTimestamp, message);
};
