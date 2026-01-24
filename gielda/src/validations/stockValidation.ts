import { check } from "express-validator";
import { validateAllowedFields } from "../utils/typeValidation";

const allowedFields = ["companyId", "userId", "amount"];

export const createStockValidation = [
  validateAllowedFields(allowedFields),
  check("companyId").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("userId").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("amount").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
];
