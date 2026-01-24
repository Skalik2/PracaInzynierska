export interface SellOfferRequest {
  companyId: number;
  userId: number;
  min_price: number;
  amount: number;
  date_limit: Date;
}
