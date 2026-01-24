import { Router } from "express";
import { createSellOffer } from "../controllers/sellOfferController";
import { createSellOfferValidation } from "../validations/sellOfferValidation";

export const sellOfferRouter = Router();

sellOfferRouter.post("/create", createSellOfferValidation, createSellOffer);

