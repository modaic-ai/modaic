# Table of Contents

* [text](#modaic.context.text)
  * [LongText](#modaic.context.text.LongText)
    * [chunk\_text](#modaic.context.text.LongText.chunk_text)

<a id="modaic.context.text"></a>

# text

<a id="modaic.context.text.LongText"></a>

## LongText Objects

```python
class LongText(Molecular)
```

<a id="modaic.context.text.LongText.chunk_text"></a>

#### chunk\_text

```python
def chunk_text(
        chunk_fn: Callable[[str], List[str | tuple[str, dict]]]) -> List[Text]
```

Chunk the text into smaller Context objects.

**Arguments**:

- `chunk_fn` - A function that takes in a string and returns a list of strings or string-metadata pairs.
  

**Returns**:

  A list of Context objects.

