import Redis from "ioredis";
import dotenv from "dotenv";

dotenv.config();

const redis = new Redis({
  host: process.env.REDIS_HOST || "localhost",
  port: 6379,
});

export const banUser = async (userId: number, durationSeconds: number = 300) => {
  await redis.set(`ban:user:${userId}`, "true", "EX", durationSeconds);
  console.log(`[SECURITY] User ${userId} has been BANNED for ${durationSeconds}s`);
};

export const isUserBanned = async (userId: number): Promise<boolean> => {
  const result = await redis.get(`ban:user:${userId}`);
  return result === "true";
};

export default redis;