from modaic.precompiled_agent import PrecompiledConfig, PrecompiledAgent
from typing import Type, Optional
import dspy
from .indexer import TableRagIndexer


# Signatures
class NL2SQL(dspy.Signature):
    """You are an expert in SQL and can generate SQL statements based on table schemas and query requirements.
    Respond as concisely as possible, providing only the SQL statement without any additional explanations."""

    schema_list = dspy.InputField(
        desc="Based on the schemas please use MySQL syntax to the user's query"
    )
    user_query = dspy.InputField(desc="The user's query")
    sql = dspy.OutputField(
        desc="Please wrap the generated SQL statement with ```sql ```, and warp table name and each column name metioned in sql with ``, for example: ```sql SELECT `name` FROM `student_sheet1` WHERE `age` > '15';"
    )


class Main(dspy.Signature):
    """
    Next, you will complete a table-related question answering task. Based on the provided materials such as the table content (in Markdown format), you need to analyze the User Query.
    And try to decide whether the User Input Query should be broken down into subqueries. You are provided with "solve_subquery" tool that can get answer for the subqueries.
    After you have collected sufficient information, you need to generate comprehensive answers.

    Instructions:
    1. Carefully analyze each user query through step-by-step reasoning.
    2. If the query needs information more than the given table content：
        - Decompose the query into subqueries.
        - Process one subquery at a time.
        - Use "solve_subquery" tool to get answers for each subquey.
    3. If a query can be answered by table content, do not decompose it. And directly put the orignal query into the "solve_subquery" tool.
        The "solve_subquery" tool utilizes SQL execution inside, it can solve complex subquery on table through one tool call.
    4. Generate exactly ONE subquery at a time.
    5. Write out all terms completely - avoid using abbreviations.
    6. When you have sufficient information, provide the final answer in the following format:
        <Answer>: [your complete response]
    Please start!
    """

    table_content = dspy.InputField()
    user_input_query = dspy.InputField()
    answer = dspy.OutputField()


class TableRAGConfig(PrecompiledConfig):
    agent_type = "TableRAGAgent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TableRAGAgent(PrecompiledAgent):
    config_class = TableRAGConfig

    def __init__(self, config: TableRAGConfig, indexer: TableRagIndexer, **kwargs):
        super().__init__(config, **kwargs)
        nl2sql = dspy.Predict(NL2SQL)
        self.nl2sql = dspy.Refine(
            module=nl2sql, N=3, reward_fn=one_word_answer, threshold=1.0
        )

        self.indexer = indexer

    def forward(self, user_query: str, table_id: Optional[str] = None) -> str:
        if table_id is not None:
            user_query = user_query + f"The given table is in {table_id}"
        else:
            user_query = user_query

        return self.nl2sql(
            schema=self.indexer.sql_db.get_table_schema(), user_query=user_query
        )
