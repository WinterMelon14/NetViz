import sys


def fail_during_class_definition(selected_class):
    print("diagnostic" * 200_000, file=sys.stderr)
    raise RuntimeError("fixture import failure after excessive diagnostics")


@fail_during_class_definition
class UserModel:
    pass
