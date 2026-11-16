from modaic import AsyncModaic, Modaic


def test_prediction_resource_aliases() -> None:
    client = Modaic(api_key="test")
    assert client.predictions is client.decisions
    assert client.batch_predictions is client.batch_decisions
    client.close()


async def test_async_prediction_resource_aliases() -> None:
    client = AsyncModaic(api_key="test")
    assert client.predictions is client.decisions
    assert client.batch_predictions is client.batch_decisions
    await client.close()
