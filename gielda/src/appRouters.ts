import { companyRouter } from "./routes/companyRouter";
import { stockRateRouter } from "./routes/stockRateRouter";
import { stockRouter } from "./routes/stockRouter";
import { userRouter } from "./routes/userRouter";
import { buyOfferRouter } from "./routes/buyOfferRouter";
import { sellOfferRouter } from "./routes/sellOfferRouter";
import { transactionRouter } from "./routes/transactionRouter";

export const APPROUTER = [
  { path: "user", router: userRouter },
  { path: "company", router: companyRouter },
  { path: "stock", router: stockRouter },
  { path: "stockrate", router: stockRateRouter },
  { path: "buyOffer", router: buyOfferRouter },
  { path: "sellOffer", router: sellOfferRouter },
  { path: "transaction", router: transactionRouter },
];
