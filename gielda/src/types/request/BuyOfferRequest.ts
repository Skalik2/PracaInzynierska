export interface BuyOfferRequest {
  companyId: number;
  userId: number;
  max_price: number;
  amount: number;
  date_limit: Date;
}
