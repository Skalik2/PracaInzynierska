import { validationResult } from "express-validator";
import { AppError } from "../utils/appError";
import { TypedRequestBody } from "../utils/TypedRequestBody";
import { NextFunction, Request, Response } from "express";
import { StockRateRequest } from "../types/request/StockRateRequest";
import {
  createStockRateService,
  allActualStockRatesService,
  actualCompanyStockRateService,
} from "../services/stockRateService";
import { catchAsync } from "../utils/catchAsync";
import { recordApiActivity } from "../utils/activityMonitor";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";
import { APILog } from "../database/logDB/services/addLog";

export const allActualStockRates = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const { result, databaseTime } = await allActualStockRatesService();
  const end = new Date();

  res.json({ result });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const userIdHeader = req.headers["x-user-id"];
  const userId = userIdHeader ? Number(userIdHeader) : undefined;

  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/stockrate${req.path}`,
    requestId,
    userPersona,
    userId: userId,
  };

  recordApiActivity(ApiLog);
});

export const actualCompanyStockRate = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const companyId: any = req.params.id;
  const { result, databaseTime } = await actualCompanyStockRateService(companyId);
  const end = new Date();

  res.json(result);

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const userIdHeader = req.headers["x-user-id"];
  const userId = userIdHeader ? Number(userIdHeader) : undefined;
  
  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/stockrate${req.path}`,
    requestId,
    userPersona,
    userId: userId,
  };

  recordApiActivity(ApiLog);
});

export const createStockRate = catchAsync(
  async (req: TypedRequestBody<StockRateRequest>, res: Response, next: NextFunction) => {
    const start = new Date();
    const requestId: string = Array.isArray(req.headers["x-request-id"])
      ? req.headers["x-request-id"][0]
      : req.headers["x-request-id"] || "default-request-id";
    const errors = validationResult(req);

    if (!errors.isEmpty()) {
      return next(new AppError("Validation errors", 400, errors.array()));
    }

    const { result, databaseTime } = await createStockRateService(req.body);
    const end = new Date();

    res.json({ message: "Stock added", result });

    const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
    const ApiLog: APILog = {
      apiMethod: ApiMethod.POST,
      applicationTime: new Date().getTime() - start.getTime(),
      databaseTime,
      endpointUrl: `/stockrate${req.path}`,
      requestId,
      userPersona,
    };

    recordApiActivity(ApiLog);
  }
);
