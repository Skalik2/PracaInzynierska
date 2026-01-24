import { NextFunction, Request, Response } from "express";
import { TypedRequestBody } from "../utils/TypedRequestBody";
import { validationResult } from "express-validator";
import { AppError } from "../utils/appError";
import { SellOfferRequest } from "../types/request/SellOfferRequest";
import { createSellOfferService, deleteSellOfferService } from "../services/sellOfferService";
import { catchAsync } from "../utils/catchAsync";
import { recordApiActivity } from "../utils/activityMonitor";
import { APILog } from "../database/logDB/services/addLog";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";

export const createSellOffer = catchAsync(
  async (req: TypedRequestBody<SellOfferRequest>, res: Response, next: NextFunction) => {
    const start = new Date();
    const requestId: string = Array.isArray(req.headers["x-request-id"])
      ? req.headers["x-request-id"][0]
      : req.headers["x-request-id"] || "default-request-id";
    const error = validationResult(req);
    if (!error.isEmpty()) return next(new AppError("Validation errors", 400, error.array()));

    const { result, databaseTime } = await createSellOfferService(req.body);
    const end = new Date();

    res.json({ message: "Sell offer added", result });

    const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
    const ApiLog: APILog = {
      apiMethod: ApiMethod.POST,
      applicationTime: new Date().getTime() - start.getTime(),
      databaseTime,
      endpointUrl: `/selloffer${req.path}`,
      requestId,
      userId: req.body.userId,
      userPersona,
    };

    recordApiActivity(ApiLog);
  }
);

export const deleteSellOffer = catchAsync(async (req: Request, res: Response, next: NextFunction) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const sellOfferId: any = req.params.id;
  const databaseTime = await deleteSellOfferService(sellOfferId);
  const end = new Date();

  res.status(200).json({ message: "Sell offer deleted successfully" });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.DELETE,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/selloffer${req.path}`,
    requestId,
    userId: req.body.userId,
    userPersona,
  };

  recordApiActivity(ApiLog);
});

