import { BaseEntity, Column, Entity, ManyToOne, OneToMany, PrimaryGeneratedColumn, RelationId } from "typeorm";
import { User } from "./UsesEntitie";
import { Company } from "./CompanyEntities";
import { TransactionD } from "./TransactionEntitie";

@Entity()
export class BuyOffer extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @ManyToOne(() => Company, (company) => company.buyOffers)
  company: Company;

  @Column("decimal", { scale: 2 })
  max_price: number;

  @Column()
  start_amount: number;

  @Column()
  date_limit: Date;

  @Column()
  actual: boolean;

  @Column()
  amount: number;

  @OneToMany(() => TransactionD, (TransactionD) => TransactionD.buyOffer)
  transactions: TransactionD[];

  @ManyToOne(() => User, (user) => user.buyOffers)
  user: User;

  @RelationId((buyOffer: BuyOffer) => buyOffer.user)
  userId: number;
}
