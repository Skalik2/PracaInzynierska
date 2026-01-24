import { Router } from "express";
import { createStockRateValidation } from "../validations/stockRateValidation";
import { createStockRate, allActualStockRates, actualCompanyStockRate } from "../controllers/stockRateController";

export const stockRateRouter = Router();

stockRateRouter.post("/create", createStockRateValidation, createStockRate);
stockRateRouter.get("/all", allActualStockRates);
stockRateRouter.get("/company/:id", actualCompanyStockRate);
