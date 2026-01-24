import { Request, Response, NextFunction } from "express";
import { AppError } from "../utils/appError";

export const validateAllowedFields = (allowedFields: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const keys = Object.keys(req.body);
    for (const key of keys) {
      if (!allowedFields.includes(key)) {
        return next(new AppError(`'${key}' is not allowed`, 400, []));
      }
    }
    next();
  };
};
