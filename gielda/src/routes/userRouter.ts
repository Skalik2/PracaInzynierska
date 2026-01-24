import { Router } from "express";
import { addUserMoney, allUsers, createUser, deleteUser, getUser } from "../controllers/userController";
import { createUserValidation } from "../validations/userValidation";

export const userRouter = Router();

userRouter.get("/allusers", allUsers);
userRouter.get("/:id", getUser);
userRouter.post("/create", createUserValidation, createUser);
userRouter.delete("/delete/:id", deleteUser);
userRouter.post("/money", addUserMoney);
