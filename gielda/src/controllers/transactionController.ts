import { Request, Response } from "express";
import { allTransactionsService } from "../services/transactionService";
import { catchAsync } from "../utils/catchAsync";
import { APILog } from "../database/logDB/services/addLog";
import { recordApiActivity } from "../utils/activityMonitor";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";

export const allTransactions = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId = req.headers["x-request-id"] as string || "default";
  
  const transactions = await allTransactionsService(req, res);
  res.json({ transactions });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const userIdHeader = req.headers["x-user-id"];
  const userId = userIdHeader ? Number(userIdHeader) : undefined;

  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime: 0, 
    endpointUrl: `/transaction${req.path}`,
    requestId,
    userPersona,
    userId,
  };
  recordApiActivity(ApiLog);
});