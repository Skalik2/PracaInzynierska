import { User } from "../entities/UsesEntitie";
import { AppError } from "../utils/appError";
import { AppDataSource } from "../database/dataSource";
import { UserRequest } from "../types/request/UserRequest";

export const createUserService = async (newUserData: UserRequest) => {
  const { name, surname, username, password, email } = newUserData;

  const start = new Date();
  const newUser = await User.save({
    name,
    surname,
    username,
    password,
    email,
    money: 0,
  });
  const end = new Date();

  return { result: newUser, databaseTime: end.getTime() - start.getTime() };
};

export const getUserService = async (userId: number) => {
  const start = new Date();
  const user = await User.findOne({ where: { id: userId } });

  if (!user) {
    throw new AppError("User not found", 404);
  }
  const end = new Date();

  return { result: user, databaseTime: end.getTime() - start.getTime() };
};

export const findAllUsersService = async () => {
  const start = new Date();
  const users = await User.find();
  const end = new Date();

  return { result: users, databaseTime: end.getTime() - start.getTime() };
};

export const deleteUserService = async (userId: number) => {
  const start = new Date();
  const user = await User.findOne({ where: { id: userId } });
  if (!user) {
    throw new AppError("User not found", 404);
  }

  await User.delete({ id: userId });
  const end = new Date();
  return end.getTime() - start.getTime();
};

export const updateUserService = async (user: User) => {
  await User.save(user);
};

export const updateUserMoney = async (userId: number, deltaMoney: number) => {
  const start = new Date();
  const entityManager = AppDataSource.manager;

  await entityManager.transaction(async (transactionalEntityManager) => {
    const user = await transactionalEntityManager.findOne(User, {
      where: { id: userId },
      lock: { mode: "pessimistic_write" },
    });

    if (user) {
      const newValue = Number(user.money) + Number(deltaMoney);
      user.money = Number(newValue.toFixed(2));
      await transactionalEntityManager.save(user);
    }
  });
  const end = new Date();
  return end.getTime() - start.getTime();
};
