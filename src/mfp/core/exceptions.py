class MFPError(Exception):
    """Base exception for MFP."""


class DataValidationError(MFPError):
    """Raised when data validation fails."""


class SplitError(MFPError):
    """Raised when temporal split constraints are violated."""


class ModelError(MFPError):
    """Raised when model operations fail."""


class ArtifactError(MFPError):
    """Raised when artifact loading/saving fails."""


class InferenceError(MFPError):
    """Raised when inference fails."""