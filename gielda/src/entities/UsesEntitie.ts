import { BaseEntity, Column, Entity, OneToMany, PrimaryGeneratedColumn } from "typeorm";
import { Stock } from "./StockEntities";
import { BuyOffer } from "./BuyOfferEntitie";
import { SellOffer } from "./SellOfferEntitie";

@Entity("user")
export class User extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column()
  surname: string;

  @Column()
  username: string;

  @Column()
  password: string;

  @Column()
  email: string;

  @Column("decimal", { scale: 2 })
  money: number;

  @OneToMany(() => Stock, (stock) => stock.user)
  stocks: Stock[];

  @OneToMany(() => BuyOffer, (buyOffer) => buyOffer.user)
  buyOffers: BuyOffer[];

  @OneToMany(() => SellOffer, (sellOffer) => sellOffer.user)
  sellOffers: SellOffer[];
}
