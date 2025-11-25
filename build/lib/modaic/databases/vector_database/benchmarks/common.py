from typing import Dict, Iterable, List, Tuple

from more_itertools import peekable
from PIL import Image

from modaic.context.base import Context, Embeddable


def _has_multiple_embedmes(
    item: Embeddable,
):
    """
    Check if the item has multiple embedmes.
    """
    return item.embedme.__code__.co_argcount == 2


def _items_have_multiple_embedmes(
    records: Iterable[Embeddable | Tuple[str | Image.Image, Context]],
):
    """
    Check if the first record has multiple embedmes.
    """
    p = peekable(records)
    first_item = p.peek()
    if isinstance(first_item, Embeddable) and _has_multiple_embedmes(first_item):
        return True
    return False


def _add_item_embedme(
    embedmes: Dict[str | None, List[str | Image.Image]],
    item: Embeddable | Tuple[str | Image.Image, Context],
):
    """
    Adds an item to the embedmes dictionary.
    """
    # Fast type check for tuple
    if type(item) is tuple:
        embedme = item[0]
        for index in embedmes.keys():
            embedmes[index].append(embedme)
    elif _has_multiple_embedmes(item):
        # CAVEAT: Context objects that implement Embeddable protocol and take in an index name as a parameter also accept None as the default index.
        for index in embedmes.keys():
            embedmes[index].append(item.embedme(index))
    else:
        embedmes[None].append(item.embedme())
