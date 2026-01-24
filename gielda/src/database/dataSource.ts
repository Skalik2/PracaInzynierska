import dotenv from "dotenv";
import { DataSource } from "typeorm";
import { createDatabase, dropDatabase } from "typeorm-extension";
import { User } from "../entities/UsesEntitie";
import { Stock } from "../entities/StockEntities";
import { BuyOffer } from "../entities/BuyOfferEntitie";
import { Company } from "../entities/CompanyEntities";
import { SellOffer } from "../entities/SellOfferEntitie";
import { StockRate } from "../entities/StockRateEntitie";
import { TransactionD } from "../entities/TransactionEntitie";

dotenv.config({ path: `${process.cwd()}/./.env` });

const entities = [BuyOffer, Company, SellOffer, Stock, StockRate, TransactionD, User];

export const AppDataSource = new DataSource({
  type: "postgres",
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT),
  username: process.env.DB_USERNAME,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  entities,
  synchronize: true,
  logging: false,
  ssl: false,
});

export async function initializeDatabase() {
  await dropDatabase({ options: AppDataSource.options });
  await createDatabase({ options: AppDataSource.options });
  await AppDataSource.initialize();
}
