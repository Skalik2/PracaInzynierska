import { Router } from "express";
import { allTransactions } from "../controllers/transactionController";

export const transactionRouter = Router();

transactionRouter.get("/all", allTransactions);
