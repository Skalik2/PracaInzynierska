import { Router } from "express";
import { createBuyOfferValidation } from "../validations/buyOfferValidation";
import { createBuyOffer, deleteBuyOffer, usersBuyOffer } from "../controllers/buyOfferController";

export const buyOfferRouter = Router();

buyOfferRouter.get("/user/:id", usersBuyOffer);
buyOfferRouter.post("/create", createBuyOfferValidation, createBuyOffer);
buyOfferRouter.post("/delete/:id", deleteBuyOffer);
