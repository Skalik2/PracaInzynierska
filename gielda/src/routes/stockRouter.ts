import { Router } from "express";
import { createStock, getStock, getStockByUserId } from "../controllers/stockController";
import { createStockValidation } from "../validations/stockValidation";

export const stockRouter = Router();

stockRouter.post("/create", createStockValidation, createStock);
stockRouter.get("/:id", getStock);
stockRouter.get("/user/:id", getStockByUserId);
