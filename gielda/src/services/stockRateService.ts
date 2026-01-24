import { StockRateRequest } from "../types/request/StockRateRequest";
import { Company } from "../entities/CompanyEntities";
import { AppError } from "../utils/appError";
import { StockRate } from "../entities/StockRateEntitie";

export const allActualStockRatesService = async () => {
  const start = new Date();
  const stockRates = await StockRate.find({ where: { actual: true } });
  const end = new Date();

  return { result: stockRates, databaseTime: end.getTime() - start.getTime() };
};

export const actualCompanyStockRateService = async (companyId: number) => {
  const start = new Date();
  const stockRate = await StockRate.findOne({ where: { actual: true, company: { id: companyId } } });
  const end = new Date();

  return { result: stockRate?.rate, databaseTime: end.getTime() - start.getTime() };
};

export const createStockRateService = async (newStockRateDate: StockRateRequest) => {
  const start = new Date();
  const { companyId, rate } = newStockRateDate;

  // Walidacja wartości kursu
  if (rate <= 0) {
    throw new AppError("Invalid stock rate: Rate must be greater than 0", 400);
  }

  const company = await Company.findOne({ where: { id: companyId } });

  if (!company) {
    throw new AppError("Company not found", 404);
  }

  const newStockRate = StockRate.save({
    company,
    date_inc: new Date().toJSON(),
    actual: true,
    rate,
  });
  const end = new Date();

  return { result: newStockRate, databaseTime: end.getTime() - start.getTime() };
};

export const updateStockRateByCompanyIdService = async (stockRateDate: StockRateRequest) => {
  const start = new Date();
  const { companyId, rate } = stockRateDate;

  // Walidacja wartości kursu
  if (rate <= 0) {
    throw new AppError("Invalid stock rate: Rate must be greater than 0", 400);
  }

  const company = await Company.findOne({ where: { id: companyId } });

  if (!company) {
    throw new AppError("Company not found", 404);
  }

  // Ustawienie aktualnego kursu na nieaktualny
  await StockRate.update({ company, actual: true }, { actual: false });

  const newStockRate = StockRate.save({
    company,
    date_inc: new Date().toISOString(),
    actual: true,
    rate,
  });
  const end = new Date();

  return end.getTime() - start.getTime();
};
