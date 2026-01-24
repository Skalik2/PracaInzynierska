import { BuyOffer } from "../../entities/BuyOfferEntitie";
import { SellOffer } from "../../entities/SellOfferEntitie";
import { BuyOfferRes } from "../response/BuyOfferRes";
import { SellOfferRes } from "../response/SellOfferRes";

export interface TransactionRequest {
  sellOffer: SellOffer | SellOfferRes;
  buyOffer: BuyOffer | BuyOfferRes;
  amount: number;
  price: number;
  transactionData: Date;
}
