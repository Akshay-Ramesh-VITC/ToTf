import import_lib

_BACKEND = None

if importlib.util.find_spec("torch"):
    _BACKEND = "torch"
elif importlib.util.find_spec("tensorflow"):
    _BACKEND = "tensorflow"