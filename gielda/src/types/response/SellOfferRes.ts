export interface SellOfferRes {
  id: number;
  min_price: number;
  amount: number;
  actual: boolean;
  start_amount: number;
  date_limit: Date;
  userId: number;
}
