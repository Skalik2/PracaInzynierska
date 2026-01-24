import { BaseEntity, Column, Entity, ManyToOne, OneToMany, PrimaryGeneratedColumn } from "typeorm";
import { User } from "./UsesEntitie";
import { Company } from "./CompanyEntities";
import { SellOffer } from "./SellOfferEntitie";

@Entity()
export class Stock extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @ManyToOne(() => User, (user) => user.stocks)
  user: User;

  @ManyToOne(() => Company, (company) => company.stocks)
  company: Company;

  @OneToMany(() => SellOffer, (sellOffer) => sellOffer.user)
  sellOffers: SellOffer[];

  @Column()
  amount: number;
}
