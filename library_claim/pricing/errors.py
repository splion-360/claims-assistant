"""Name the one error every price source raises when it has no usable answer."""


class PriceUnavailable(RuntimeError):
    """Raised when a source returns no usable price. The message is safe to show.

    Attributes:
        transient: True when trying again later may succeed (timeout, rate limit,
            server error); False when the source answered definitely.
    """

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient
