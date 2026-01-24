import { check } from "express-validator";
import { validateAllowedFields } from "../utils/typeValidation";

const allowedFields = ["companyId", "userId", "min_price", "amount", "date_limit"];

export const createSellOfferValidation = [
  validateAllowedFields(allowedFields),
  check("companyId").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("userId").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("min_price").exists().withMessage("add rate").isDecimal().withMessage("Is not decimal"),
  check("amount").exists().withMessage("add companyId").isInt().withMessage("Is not int"),
  check("date_limit").exists().withMessage("add companyId").isISO8601().withMessage("Is not date"),
];
