export class AppError extends Error {
  constructor(public message: string, public statusCode: number, public errors: any[] = []) {
    super(message);
    this.statusCode = statusCode;
    this.status = `${statusCode}`.startsWith("4") ? "fail" : "error";
    this.errors = errors;

    Error.captureStackTrace(this, this.constructor);
  }

  public status: string;
}
