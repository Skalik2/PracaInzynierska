import { Entity } from "typeorm";
import { Cpu } from "./CpuEntities";

@Entity()
export class TrafficCpu extends Cpu {}
