import { check } from "express-validator";
import { validateAllowedFields } from "../utils/typeValidation";

const allowedFields = ["name", "surname", "username", "password", "email"];

export const createUserValidation = [
  validateAllowedFields(allowedFields),
  check("name").exists().withMessage("add name"),
  check("surname").exists().withMessage("add surname"),
  check("username").exists().withMessage("add username"),
  check("password").exists().withMessage("add surname"),
  check("email").isEmail().withMessage("add email"),
];
