import { Router } from "express";
import { createCompanyValidation } from "../validations/companyValidation";
import { allCompanies, createCompany, deleteCompany, getCompany } from "../controllers/companyController";

export const companyRouter = Router();

companyRouter.get("/allcompany", allCompanies);
companyRouter.get("/:id", getCompany);
companyRouter.post("/create", createCompanyValidation, createCompany);
companyRouter.post("/delete/:id", deleteCompany);
