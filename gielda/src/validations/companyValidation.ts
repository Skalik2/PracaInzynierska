import { check } from "express-validator";
import { validateAllowedFields } from "../utils/typeValidation";

const allowedFields = ["name"];

export const createCompanyValidation = [validateAllowedFields(allowedFields), check("name").exists()];
