import { BaseEntity, Column, PrimaryGeneratedColumn } from "typeorm";

export class Cpu extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "timestamp", precision: 3 })
  timestamp: Date;

  @Column("decimal", { scale: 2 })
  cpuUsage: number;

  @Column("decimal", { scale: 2 })
  memoryUsage: number;
}
