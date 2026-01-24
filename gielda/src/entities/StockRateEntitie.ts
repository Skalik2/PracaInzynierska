import { BaseEntity, Column, Entity, ManyToOne, PrimaryGeneratedColumn } from "typeorm";
import { Company } from "./CompanyEntities";

@Entity()
export class StockRate extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @ManyToOne(() => Company, (company) => company.stockRates)
  company: Company;

  @Column()
  date_inc: Date;

  @Column()
  actual: boolean;

  @Column("decimal", { scale: 2 })
  rate: number;
}
