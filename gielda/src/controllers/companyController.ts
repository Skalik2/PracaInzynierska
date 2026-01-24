import { NextFunction, Request, Response } from "express";
import { validationResult } from "express-validator";
import { AppError } from "../utils/appError";
import {
  createCompanyService,
  deleteCompanyService,
  findAllCompaniesService,
  getCompanyService,
} from "../services/companyService";
import { CompanyRequest } from "../types/request/CompanyRequest";
import { TypedRequestBody } from "../utils/TypedRequestBody";
import { catchAsync } from "../utils/catchAsync";
import { recordApiActivity } from "../utils/activityMonitor";
import { APILog } from "../database/logDB/services/addLog";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";

export const allCompanies = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const { result, databaseTime } = await findAllCompaniesService();
  const end = new Date();

  res.json(result);

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const userIdHeader = req.headers["x-user-id"];
  const userId = userIdHeader ? Number(userIdHeader) : undefined;

  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/company${req.path}`,
    requestId,
    userPersona,
    userId: userId,
  };

  recordApiActivity(ApiLog);
});

export const getCompany = catchAsync(async (req: Request, res: Response, next: NextFunction) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const companyId: any = req.params.id;

  const { result, databaseTime } = await getCompanyService(companyId);
  const end = new Date();

  res.json(result);

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const userIdHeader = req.headers["x-user-id"];
  const userId = userIdHeader ? Number(userIdHeader) : undefined;

  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/company${req.path}`,
    requestId,
    userPersona,
    userId: userId,
  };

  recordApiActivity(ApiLog);
});

export const createCompany = catchAsync(async (req: TypedRequestBody<CompanyRequest>, res: any, next: NextFunction) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return next(new AppError("Validation errors", 400, errors.array()));
  }

  const { result, databaseTime } = await createCompanyService(req.body);
  const end = new Date();

  res.json({ message: "Company added", result });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.POST,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/company${req.path}`,
    requestId,
    userPersona,
  };

  recordApiActivity(ApiLog);
});

export const deleteCompany = catchAsync(async (req: Request, res: Response, next: NextFunction) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const companyId: any = req.params.id;

  const databaseTime = await deleteCompanyService(companyId);
  const end = new Date();

  res.status(200).json({ message: "Company deleted successfully" });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.DELETE,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/company${req.path}`,
    requestId,
    userPersona,
  };

  recordApiActivity(ApiLog);
});
