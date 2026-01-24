export interface BuyOfferRes {
  id: number;
  max_price: number;
  amount: number;
  actual: boolean;
  start_amount: number;
  date_limit: Date;
  userId: number;
}
