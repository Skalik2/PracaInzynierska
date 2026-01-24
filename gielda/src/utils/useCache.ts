import NodeCache from "node-cache";
import dotenv from "dotenv";

dotenv.config({ path: `${process.cwd()}/./.env` });

const cache = new NodeCache({ stdTTL: Number(process.env.CACHE_TIME) });

export const getCache = <T>(prefix: string): T[] => {
  const keys = cache.keys().filter((key) => key.startsWith(prefix));
  return keys.map((key) => cache.get<T>(key)).filter((item) => item !== undefined) as T[];
};

export const setCache = <T>(key: string, data: T): void => {
  cache.set(key, data);
};

export const removeCache = (key: string): void => {
  cache.del(key);
};
