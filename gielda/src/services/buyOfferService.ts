import { User } from "../entities/UsesEntitie";
import { AppError } from "../utils/appError";
import { BuyOfferRequest } from "../types/request/BuyOfferRequest";
import { Company } from "../entities/CompanyEntities";
import { BuyOffer } from "../entities/BuyOfferEntitie";
import { BuyOfferRes } from "../types/response/BuyOfferRes";
import { In, Not, Raw } from "typeorm";
import { updateUserMoney, updateUserService } from "./userService";

export const createBuyOfferService = async (newBuyOfferData: BuyOfferRequest) => {
  const start = new Date();
  const { companyId, userId, max_price, amount, date_limit } = newBuyOfferData;

  const user = await User.findOne({ where: { id: userId } });
  const company = await Company.findOne({ where: { id: companyId } });

  if (!user || !company) {
    throw new AppError("User or Company not found", 404);
  }

  const estimatedPrice = Number(-1 * (amount * +max_price));

  if (+user.money - +estimatedPrice < 0) {
    throw new AppError("Not enough money", 402);
  }

  await updateUserMoney(userId, estimatedPrice);

  const newBuyOffer = await BuyOffer.save({
    user,
    company,
    max_price,
    amount,
    start_amount: amount,
    actual: true,
    date_limit,
  });

  await updateUserService(user);
  const end = new Date();

  return { result: newBuyOffer, databaseTime: end.getTime() - start.getTime() };
};

export const deleteBuyOfferService = async (buyOfferId: number) => {
  const start = new Date();
  const buyOffer = await BuyOffer.findOne({ where: { id: buyOfferId }, relations: { user: true } });
  if (!buyOffer) {
    throw new AppError("Delete offer not found", 404);
  }
  const estimatedPrice = Number(+buyOffer.amount * +buyOffer.max_price);

  await updateUserMoney(buyOffer.user.id, estimatedPrice);
  await BuyOffer.delete({ id: buyOfferId });
  const end = new Date();

  return end.getTime() - start.getTime();
};

export const usersBuyOfferService = async (userId: number) => {
  const start = new Date();
  const user = await User.findOne({ where: { id: userId } });

  if (!user) {
    throw new AppError("User not found", 404);
  }

  const buyOffers = await BuyOffer.find({ where: { user } });
  const end = new Date();

  return { result: buyOffers, databaseTime: end.getTime() - start.getTime() };
};

export const buyOffersToTradeService = async (
  companyId: number,
  skipIds: number[],
  recordsNumber: number = 100
): Promise<BuyOfferRes[]> => {
  const buyOffers = await BuyOffer.find({
    where: { actual: true, company: { id: companyId }, id: Not(In(skipIds)) },
    order: { id: "ASC", max_price: "DESC" },
    take: recordsNumber,
  });

  if (!buyOffers) {
    throw new AppError("BuyOffers not found", 404);
  }

  return buyOffers;
};

export const updateBuyOfferService = async (buyOffer: any) => {
  const start = new Date();
  await BuyOffer.save(buyOffer);
  const end = new Date();

  return end.getTime() - start.getTime();
};

export const removeExpiredBuyOffersService = async (companyId: number) => {
  const expiredOffers = await BuyOffer.find({
    where: { actual: true, date_limit: Raw((date) => `${date} < NOW()`), company: { id: companyId } },
    relations: { user: true },
  });

  if (expiredOffers.length > 0) {
    for (let i = 0; i < expiredOffers.length; i++) {
      const money = Number(expiredOffers[i].amount * +expiredOffers[i].max_price);
      expiredOffers[i].actual = false;
      await updateUserMoney(expiredOffers[i].user.id, money);
      await expiredOffers[i].save();
    }
  }
};

export const isBuyOfferExistService = async (buyOfferId: number): Promise<boolean> => {
  const offer = await BuyOffer.findOne({
    where: { id: buyOfferId },
  });

  return !!offer;
};
