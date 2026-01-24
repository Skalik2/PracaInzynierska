import { BaseEntity, Column, Entity, PrimaryGeneratedColumn } from "typeorm";

@Entity()
export class BlockedUserLog extends BaseEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: "timestamp", precision: 3 })
  timestamp: Date;

  @Column({ type: "int" })
  userId: number;

  @Column()
  endpointUrl: string;

  @Column()
  apiMethod: string;

  @Column({ nullable: true })
  reason: string;

  @Column({ nullable: true })
  userAgent: string;
}