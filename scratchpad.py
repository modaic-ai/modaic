from modaic.context import Text, LongText

query = (Text.text == "Hello, world!") & (Text.metadata["doc"] == 1)
