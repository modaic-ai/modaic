from modaic.context import Context, Relation


class TestContext(Context):
    name: str
    age: int


class TestRelation(Relation):
    prop1: str
    prop2: int


def test_create_context_to_gqlalchemy():
    pass


def test_save_node():
    pass


def test_relation_to_gqlalchemy():
    pass


def test_inline_relation():
    pass


def test_save_relation():
    pass


def test_shift_operators():
    """
    Test a few examples of using >> and << on Relation objects. Ensure the correct errors are raised when used incorrectly.
    """
    pass
