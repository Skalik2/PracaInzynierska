import { NextFunction, Request, Response } from "express";
import { TypedRequestBody } from "../utils/TypedRequestBody";
import { BuyOfferRequest } from "../types/request/BuyOfferRequest";
import { validationResult } from "express-validator";
import { AppError } from "../utils/appError";
import { createBuyOfferService, deleteBuyOfferService, usersBuyOfferService } from "../services/buyOfferService";
import { catchAsync } from "../utils/catchAsync";
import { recordApiActivity } from "../utils/activityMonitor";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";
import { APILog } from "../database/logDB/services/addLog";

export const createBuyOffer = catchAsync(
  async (req: TypedRequestBody<BuyOfferRequest>, res: Response, next: NextFunction) => {
    const start = new Date();
    const error = validationResult(req);
    if (!error.isEmpty()) {
      return next(new AppError("Validation  errors", 400, error.array()));
    }
    const { result, databaseTime } = await createBuyOfferService(req.body);
    const end = new Date();

    res.json({ message: "Buy offer added", result });

    const requestId: string = Array.isArray(req.headers["x-request-id"])
      ? req.headers["x-request-id"][0]
      : req.headers["x-request-id"] || "default-request-id";

    const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
    const ApiLog: APILog = {
      apiMethod: ApiMethod.POST,
      applicationTime: new Date().getTime() - start.getTime(),
      databaseTime,
      endpointUrl: `/buyoffer${req.path}`,
      requestId,
      userId: req.body.userId,
      userPersona: userPersona,
    };

    recordApiActivity(ApiLog);
  }
);

export const deleteBuyOffer = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const buyOfferId: any = req.params.id;

  const databaseTime = await deleteBuyOfferService(buyOfferId);
  const end = new Date();

  res.status(200).json({ message: "Buy offer deleted successfully" });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.DELETE,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/buyoffer${req.path}`,
    requestId,
    userPersona,
  };

  recordApiActivity(ApiLog);
});

export const usersBuyOffer = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const userId: any = req.params.id;

  const { result, databaseTime } = await usersBuyOfferService(userId);
  const end = new Date();

  res.json({ result });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/buyoffer${req.path}`,
    requestId,
    userId: Number(userId),
    userPersona,
  };

  recordApiActivity(ApiLog);
});
