import { BaseEntity, Column, Entity, PrimaryGeneratedColumn } from "typeorm";

@Entity()
export class TradeLog extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "timestamp", precision: 3 })
  timestamp: Date;

  @Column()
  applicationTime: number;

  @Column()
  databaseTime: number;

  @Column()
  numberOfSellOffers: number;

  @Column()
  numberOfBuyOffers: number;
}
