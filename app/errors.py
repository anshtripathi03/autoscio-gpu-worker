class JobFailure(Exception):
    """A job failed for a reason worth showing to the backend verbatim.

    `retryable` tells the backend whether resubmitting (with fresh presigned URLs)
    can succeed — e.g. an expired URL or a GPU out-of-memory is retryable, a
    corrupt voice sample is not.
    """

    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable
