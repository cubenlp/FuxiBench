"""
Custom exceptions for the Xinyun package
"""


class XinyunError(Exception):
    """Base exception for the Xinyun package."""
    pass


class PinyinError(XinyunError):
    """Exception raised when pinyin processing fails."""
    pass


class ConversionError(XinyunError):
    """Exception raised when poem conversion fails."""
    pass


class ValidationError(XinyunError):
    """Exception raised when poem validation fails."""
    pass


class InputError(XinyunError):
    """Exception raised when input is invalid."""
    pass