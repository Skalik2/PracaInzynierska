import { BaseEntity, Column, Entity, ManyToOne, OneToMany, PrimaryGeneratedColumn, RelationId } from "typeorm";
import { Company } from "./CompanyEntities";
import { User } from "./UsesEntitie";
import { Stock } from "./StockEntities";
import { TransactionD } from "./TransactionEntitie";

@Entity()
export class SellOffer extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column("decimal", { scale: 2 })
  min_price: number;

  @Column()
  start_amount: number;

  @Column()
  date_limit: Date;

  @Column()
  actual: boolean;

  @Column()
  amount: number;

  @OneToMany(() => TransactionD, (TransactionD) => TransactionD.sellOffer)
  transactions: TransactionD[];

  @ManyToOne(() => Stock, (stock) => stock.sellOffers)
  stock: Stock;

  @ManyToOne(() => User, (user) => user.sellOffers)
  user: User;

  @RelationId((sellOffer: SellOffer) => sellOffer.user)
  userId: number;
}
