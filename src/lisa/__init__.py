"""LISA: Linear, Interpretable, Slim and Aware Framework for MEG/EEG Decoding."""

__version__ = '0.1.0'


# Lazy import to avoid requiring torch when only importing constants/utils
def __getattr__(name: str):
    if name == 'LISA':
        from lisa.model.nn_modules import LISA

        return LISA
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['LISA', '__version__']
