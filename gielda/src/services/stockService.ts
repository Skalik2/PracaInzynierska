import { StockRequest } from "../types/request/StockRequest";
import { Stock } from "../entities/StockEntities";
import { User } from "../entities/UsesEntitie";
import { Company } from "../entities/CompanyEntities";
import { AppError } from "../utils/appError";
import { AppDataSource } from "../database/dataSource";

export const createStockService = async (newStockData: StockRequest) => {
  const start = new Date();
  const { companyId, userId, amount } = newStockData;

  const user = await User.findOne({ where: { id: userId } });
  const company = await Company.findOne({ where: { id: companyId } });

  if (!user || !company) {
    throw new AppError("User or Company not found", 404);
  }

  const stock = await Stock.findOne({ where: { user: { id: userId }, company: { id: companyId } } });

  if (!stock) {
    const newStock = Stock.save({
      user,
      company,
      amount,
    });
    const end = new Date();

    return { result: newStock, databaseTime: end.getTime() - start.getTime() };
  }
  stock.amount = Number(stock.amount) + Number(amount);

  stock.save();
  const end = new Date();

  return { result: stock, databaseTime: end.getTime() - start.getTime() };
};

export const getStockService = async (stockId: number) => {
  const stock = await Stock.findOne({ where: { id: stockId } });
  if (!stock) {
    throw new AppError("Stock not found", 404);
  }

  return stock;
};

export const getStockByUserIdService = async (userId: number) => {
  const start = new Date();
  const user = await User.findOne({ where: { id: userId } });

  if (!user) {
    throw new AppError("User not found", 404);
  }

  const stock = await Stock.find({ where: { user } });

  if (!stock) {
    throw new AppError("Stock not found", 404);
  }
  const end = new Date();

  return { result: stock, databaseTime: end.getTime() - start.getTime() };
};

export const updateStockByUserAndCompanyIdService = async (userId: number, companyId: number, amount: number) => {
  const start = new Date();
  const entityManager = AppDataSource.manager;

  await entityManager.transaction(async (transactionalEntityManager) => {
    const stock = await transactionalEntityManager.findOne(Stock, {
      where: { company: { id: companyId }, user: { id: userId } },
      lock: { mode: "pessimistic_write" },
    });

    if (!stock) {
      await createStockService({ companyId, userId, amount });
      const end = new Date();

      return end.getTime() - start.getTime();
    }

    if (stock) {
      stock.amount = +stock.amount + amount;
      await transactionalEntityManager.save(stock);
    }
  });

  const end = new Date();

  return end.getTime() - start.getTime();
};
