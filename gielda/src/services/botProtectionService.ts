import axios from "axios";
import * as os from "os";
import { banUser } from "../utils/redis";
import { BlockedUserLog } from "../database/logDB/entities/BlockedUserLogEntities";
import { APILog } from "../database/logDB/services/addLog";
import { getLastCpuUsage } from "../utils/logger/createlog";

const getMemoryUsage = () => {
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  return ((totalMem - freeMem) / totalMem) * 100;
};

export const verifyUserActivity = async (info: APILog) => {
  try {
    const serviceType = process.env.SERVICE_TYPE || 'MARKET';
    const payload = {
      apiTime: info.applicationTime + info.databaseTime, 
      applicationTime: info.applicationTime,
      databaseTime: info.databaseTime,
      memoryUsage: getMemoryUsage(),
      serviceType: serviceType,
      cpuUsage: getLastCpuUsage(),
      endpointUrl: info.endpointUrl, 
      apiMethod: info.apiMethod,
      userId: info.userId
    };

    const response = await axios.post(process.env.BOT_GUARD_URL + "/predict", payload);
    
    if (response.data.is_bot) {
        const score = response.data.score;
        console.log(`[BOT DETECTED] Score: ${score}, UserID: ${info.userId || 'UNKNOWN'}`);
        
        if (info.userId) {
            await banUser(info.userId, 600);
            
            try {
                const blockLog = new BlockedUserLog();
                blockLog.timestamp = new Date();
                blockLog.userId = info.userId;
                blockLog.endpointUrl = info.endpointUrl;
                blockLog.apiMethod = info.apiMethod;
                blockLog.reason = `BOT DETECTED (Score: ${score.toFixed(4)})`;
                blockLog.userAgent = "BotGuard/System"; 
                await blockLog.save();
            } catch (dbError) {
                console.error("Error saving block log:", dbError);
            }
        }
    }
  } catch (error: any) {
    console.error("Bot guard error:", error.message);
  }
};