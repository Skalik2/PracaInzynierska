import { createLog } from "./logger/createlog";
import { verifyUserActivity } from "../services/botProtectionService";
import { APILog } from "../database/logDB/services/addLog";

export const recordApiActivity = async (info: APILog) => {
  createLog(info, "marketLog");

//   To odkomentować, gdy bot protection będzie gotowy
  verifyUserActivity(info).catch(err => console.error("Error in bot protection:", err));
};