import { Column, Entity } from "typeorm";
import { Cpu } from "./CpuEntities";

@Entity()
export class TradeCpu extends Cpu {
  @Column()
  replicaId: number;
}
