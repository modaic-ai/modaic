"""Direct inference with a typed response. Requires MODAIC_API_KEY; makes a billable call."""

from modaic import (
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Modaic,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


class ProductDecision(DecisionResponse):
    outdoor: NoulAnswer
    department: ChoiceAnswer
    specificity: ScoreAnswer


def classify(client: Modaic) -> ProductDecision:
    return client.decisions.create(
        model="typesafe/jev-latest",
        state={"description": "Waterproof hiking boots, size 42"},
        questions={
            "outdoor": Noul(instructions="Is this product intended for outdoor use?"),
            "department": Choice(
                criteria={
                    "apparel": "Clothes and footwear",
                    "electronics": "Electronic devices",
                    "other": "Other products",
                }
            ),
            "specificity": Score(
                criteria=[
                    "No identifiable product",
                    "Product category only",
                    "Specific product with useful details",
                ]
            ),
        },
        response_model=ProductDecision,
    )


if __name__ == "__main__":
    with Modaic() as client:
        result = classify(client)
        print(result.department.choice, result.outdoor.noul, result.specificity.score)
