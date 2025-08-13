# TODO: Cast all filters to a common format (MQL)


class Filter:
    pass


class Prop:
    pass


class Expr:
    pass


Filter(Prop("field1") == "foo" & Prop("field2") * Prop("field3") > 100)
