import { BaseEntity, Column, Entity, ManyToOne, PrimaryGeneratedColumn } from "typeorm";
import { SellOffer } from "./SellOfferEntitie";
import { BuyOffer } from "./BuyOfferEntitie";

@Entity()
export class TransactionD extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @ManyToOne(() => SellOffer, (sellOffer) => sellOffer.transactions)
  sellOffer: SellOffer;

  @ManyToOne(() => BuyOffer, (buyOffer) => buyOffer.transactions)
  buyOffer: BuyOffer;

  @Column()
  amount: number;

  @Column("decimal", { scale: 2 })
  price: number;

  @Column()
  transactionData: Date;
}
