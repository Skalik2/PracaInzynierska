import { NextFunction, Request, Response } from "express";
import {
  createUserService,
  deleteUserService,
  findAllUsersService,
  getUserService,
  updateUserMoney,
} from "../services/userService";
import { validationResult } from "express-validator";
import { AppError } from "../utils/appError";
import { UserRequest } from "../types/request/UserRequest";
import { TypedRequestBody } from "../utils/TypedRequestBody";
import { catchAsync } from "../utils/catchAsync";
import { recordApiActivity } from "../utils/activityMonitor";
import { APILog } from "../database/logDB/services/addLog";
import { ApiMethod } from "../database/logDB/entities/MarketLogEntities";

export const allUsers = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId = req.headers["x-request-id"];
  const { result, databaseTime } = await findAllUsersService();
  const end = new Date();

  res.json({ result });
});

export const getUser = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const userId: any = req.params.id;

  const { result, databaseTime } = await getUserService(userId);
  const end = new Date();

  res.json(result);

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.GET,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/user${req.path}`,
    requestId,
    userId: Number(userId),
    userPersona,
  };

  recordApiActivity(ApiLog);
});

export const createUser = catchAsync(async (req: TypedRequestBody<UserRequest>, res: Response, next: NextFunction) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const error = validationResult(req);
  if (!error.isEmpty()) {
    return next(new AppError("Validation errors", 400, error.array()));
  }
  const { result, databaseTime } = await createUserService(req.body);
  const end = new Date();

  res.json({ message: "User added", result });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.POST,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/user${req.path}`,
    requestId,
    userId: result.id,
    userPersona,
  };

  recordApiActivity(ApiLog);
});

export const deleteUser = catchAsync(async (req: Request, res: Response) => {
  const start = new Date();
  const requestId: string = Array.isArray(req.headers["x-request-id"])
    ? req.headers["x-request-id"][0]
    : req.headers["x-request-id"] || "default-request-id";
  const userId: any = req.params.id;
  const databaseTime = await deleteUserService(userId);
  const end = new Date();

  res.status(200).json({ message: "User deleted successfully" });

  const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
  const ApiLog: APILog = {
    apiMethod: ApiMethod.DELETE,
    applicationTime: new Date().getTime() - start.getTime(),
    databaseTime,
    endpointUrl: `/user${req.path}`,
    requestId,
    userId: Number(userId),
    userPersona,
  };

  recordApiActivity(ApiLog);
});

export const addUserMoney = catchAsync(
  async (req: TypedRequestBody<{ userId: number; money: number }>, res: Response) => {
    const start = new Date();
    const requestId: string = Array.isArray(req.headers["x-request-id"])
      ? req.headers["x-request-id"][0]
      : req.headers["x-request-id"] || "default-request-id";
    const { userId, money } = req.body;
    const databaseTime = await updateUserMoney(userId, money);
    const end = new Date();

    res.status(200).json({ message: "Money added successfully" });

    const userPersona = req.headers["x-user-persona"] as string || "UNKNOWN";
    const ApiLog: APILog = {
      apiMethod: ApiMethod.POST,
      applicationTime: new Date().getTime() - start.getTime(),
      databaseTime,
      endpointUrl: `/user${req.path}`,
      requestId,
      userId: userId,
      userPersona,
    };

    recordApiActivity(ApiLog);
  }
);
