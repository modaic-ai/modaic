"""Typed signature inference. Requires MODAIC_API_KEY; makes a billable call."""

from typing import Literal

from modaic import Choice, Date, Modaic, Noul, Number, Question, Signature, Span


class ProductDecision(Signature):
    outdoor: Noul = Question(instructions="Is this product intended for outdoor use?")
    department: Choice[Literal["apparel", "electronics", "other"]] | None = Question(
        instructions="Which department fits this product?",
        criteria=["Clothes and footwear", "Electronic devices", "Other products"],
    )
    size: Number = Question(instructions="What is the shoe size?", min=0)
    evidence: Span = Question(instructions="Extract the product name")
    published: Date | None = Question(instructions="Publication date?", nullable=0.8)


def classify(client: Modaic) -> ProductDecision:
    return client.decisions.create(
        model="mo-latest",
        state="Waterproof hiking boots, size 42",
        questions=ProductDecision,
    )


if __name__ == "__main__":
    with Modaic() as client:
        result = classify(client)
        if result.department is not None:
            print(result.department.choice)  # Literal["apparel", "electronics", "other"]
        print(result.size.number, result.size.confidence)
        print(result.outdoor.noul, result.evidence.span.text)
        assert result.outdoor is result.answers["outdoor"]
