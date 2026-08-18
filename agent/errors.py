class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails.

    ⭐ 面向用户:消息里写的是「证书包该怎么配」。⛔ 不许被出站层压成
    「服务内部异常」—— 那会把用户唯一能照做的那句话拿掉。
    契约见 ``agent.error_classifier.USER_ACTIONABLE_ATTR``。
    """

    hermes_user_actionable = True


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config.

    ⭐ 面向用户:消息里带 ``hermes moa list`` 这类可照做的指令,⛔ 同上不许被压掉。
    """

    hermes_user_actionable = True
