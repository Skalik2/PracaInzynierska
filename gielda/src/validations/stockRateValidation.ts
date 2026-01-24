import { check } from "express-validator";
import { validateAllowedFields } from "../utils/typeValidation";

const allowedFields = ["companyId", "rate"];

export const createStockRateValidation = [
  validateAllowedFields(allowedFields),
  check("companyId").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("rate").exists().withMessage("add rate").isDecimal().withMessage("Is not decimal"),
];
