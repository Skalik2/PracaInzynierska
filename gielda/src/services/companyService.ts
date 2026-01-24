import { CompanyRequest } from "../types/request/CompanyRequest";
import { Company } from "../entities/CompanyEntities";
import { AppError } from "../utils/appError";

export const findAllCompaniesService = async () => {
  const start = new Date();
  const company = await Company.find();
  const end = new Date();

  return { result: company, databaseTime: end.getTime() - start.getTime() };
};

export const getCompanyService = async (companyId: number) => {
  const start = new Date();
  const company = await Company.findOne({ where: { id: companyId } });

  if (!company) {
    throw new AppError("Company not found", 404);
  }
  const end = new Date();

  return { result: company, databaseTime: end.getTime() - start.getTime() };
};

export const getCompanysIdServices = async () => {
  const companysId = await Company.find();

  if (!companysId.length) {
    return [];
  }

  return companysId.map((obj) => obj.id);
};

export const createCompanyService = async (newComapnyData: CompanyRequest) => {
  const start = new Date();
  const { name } = newComapnyData;

  const newCompany = await Company.save({ name });
  const end = new Date();

  return { result: newCompany, databaseTime: end.getTime() - start.getTime() };
};

export const deleteCompanyService = async (companyId: number) => {
  const start = new Date();
  const company = await Company.findOne({ where: { id: companyId } });
  if (!company) {
    throw new AppError("Company not found", 404);
  }

  await Company.delete({ id: companyId });
  const end = new Date();

  return end.getTime() - start.getTime();
};
