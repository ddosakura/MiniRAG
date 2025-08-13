import { Response, Request } from 'httpx';

export class APIStatusError extends Error {
  response: Response;
  status_code: number;
  request_id: string | null;

  constructor(message: string, response: Response, body: any | null) {
    super(message);
    this.response = response;
    this.status_code = response.status_code;
    this.request_id = response.headers.get("x-request-id") || null;
  }
}

export class APIConnectionError extends Error {
  constructor(message: string = "Connection error.", request: Request) {
    super(message);
  }
}

export class BadRequestError extends APIStatusError {
  status_code: 400 = 400;
}

export class AuthenticationError extends APIStatusError {
  status_code: 401 = 401;
}

export class PermissionDeniedError extends APIStatusError {
  status_code: 403 = 403;
}

export class NotFoundError extends APIStatusError {
  status_code: 404 = 404;
}

export class ConflictError extends APIStatusError {
  status_code: 409 = 409;
}

export class UnprocessableEntityError extends APIStatusError {
  status_code: 422 = 422;
}

export class RateLimitError extends APIStatusError {
  status_code: 429 = 429;
}

export class APITimeoutError extends APIConnectionError {
  constructor(request: Request) {
    super("Request timed out.", request);
  }
}