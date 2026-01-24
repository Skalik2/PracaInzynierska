import { BaseEntity, Column, Entity, OneToMany, PrimaryGeneratedColumn } from "typeorm";
import { StockRate } from "./StockRateEntitie";
import { BuyOffer } from "./BuyOfferEntitie";
import { Stock } from "./StockEntities";

@Entity()
export class Company extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @OneToMany(() => StockRate, (stockRate) => stockRate.company)
  stockRates: StockRate[];

  @OneToMany(() => BuyOffer, (buyOffer) => buyOffer.company)
  buyOffers: BuyOffer[];

  @OneToMany(() => Stock, (stock) => stock.company)
  stocks: Stock[];
}
