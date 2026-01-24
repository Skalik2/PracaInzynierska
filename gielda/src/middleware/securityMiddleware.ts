import { Request, Response, NextFunction } from "express";
import { isUserBanned } from "../utils/redis";
import { AppError } from "../utils/appError";
import { BlockedUserLog } from "../database/logDB/entities/BlockedUserLogEntities";

export const securityMiddleware = async (req: Request, res: Response, next: NextFunction) => {
  const userIdParam = req.body.userId || req.params.id || req.headers["x-user-id"];

  if (userIdParam) {
    const userId = Number(userIdParam);
    
    if (!isNaN(userId)) {
        const banned = await isUserBanned(userId);
        
        if (banned) {
          console.log(`[BLOCKED] Request from banned user: ${userId}`);
          try {
            const log = new BlockedUserLog();
            log.timestamp = new Date();
            log.userId = userId;
            log.endpointUrl = req.originalUrl || req.url;
            log.apiMethod = req.method;
            log.reason = "User banned in Redis";
            log.userAgent = req.get('User-Agent') || 'Unknown';
            
            await log.save();
          } catch (error) {
            console.error(error);
          }
          return next(new AppError("Access Denied", 403));
        }
    }
  }

  next();
};