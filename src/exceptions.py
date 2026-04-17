class AppError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class PersistingError(AppError):
    def __init__(self, message: str, code: str = "PERSISTING_FAILED") -> None:
        super().__init__(message, code)


class ReadingError(AppError):
    def __init__(self, message: str, code: str = "READING_FAILED") -> None:
        super().__init__(message, code)


class UpdateError(AppError):
    def __init__(self, message: str, code: str = "UPDATE_FAILED") -> None:
        super().__init__(message, code)
