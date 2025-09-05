from modaic.precompiled_agent import PrecompiledConfig, PrecompiledAgent
from modaic.context import Table
from typing import Type, Optional
import dspy
from agent.indexer import TableRAGIndexer
import json
from modaic.databases import (
    VectorDatabase,
    MilvusBackend,
    SearchResult,
    SQLDatabase,
    SQLiteConfig,
)
import os

# import utils.google_api as google_api
# import utils.outlook_api as outlook_api
# import utils.zoom_api as zoom_api
from agent.config import TableRAGConfig


# Signatures
class NL2SQL(dspy.Signature):
    """You are an expert in SQL and can generate SQL statements based on table schemas and query requirements.
    Respond as concisely as possible, providing only the SQL statement without any additional explanations."""

    schema_list = dspy.InputField(
        desc="Based on the schemas please use MySQL syntax to the user's query"
    )
    user_query = dspy.InputField(desc="The user's query")
    answer = dspy.OutputField(desc="Answer to the user's query")


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


class SubQuerySummarizer(dspy.Signature):
    """
    You are about to complete a table-based question answernig task using the following two types of reference materials:

    Note:
    1. The markdown table content in Original Content may be incomplete.
    2. You should cross-validate the given two materials:
        - if the answers are the same, directly output the answer.
        - if the "SQL execution result" contains error or is empty, you should try to answer based on the Original Content.
        - if the two materials shows conflit, you should think about each of them, and finally give an answer.
    """

    original_content = dspy.InputField(
        desc="Content 1: Original content (table content is provided in Markdown format)"
    )
    table_schema = dspy.InputField(desc="The user given table schema")
    gnerated_sql = dspy.InputField(
        desc="SQL generated based on the schema and the user question"
    )
    sql_execute_result = dspy.InputField(desc="SQL execution results")
    user_query = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="Answer to the user's question")


class TableRAGAgent(PrecompiledAgent):
    config_class = TableRAGConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = dspy.ReAct(Main, tools=[self.solve_subquery])
        self.nl2sql = dspy.ReAct(NL2SQL, tools=[self.indexer.sql_query])
        self.subquery_summarizer = dspy.Predict(SubQuerySummarizer)
        self.set_lm(dspy.LM("openai/gpt-4o-mini"))

    def forward(self, user_query: str, table_id: Optional[str] = None, **kwargs) -> str:
        if table_id is not None:
            self.user_query = user_query + f"The given table is in {table_id}"
        else:
            self.user_query = user_query
        print("USER QUERY", self.user_query)
        related_table_serialized = self.indexer.retrieve(
            self.user_query,
            k_recall=self.config.k_recall,
            k_rerank=self.config.k_rerank,
            type="table",
        )[0]  # TODO: handle multiple tables
        print("RELATED TABLE", related_table_serialized)
        related_table = self.indexer.get_table(
            related_table_serialized.metadata["schema"]["table_name"]
        )
        self.table_md = related_table.markdown()
        self.table_schema = json.dumps(related_table.metadata["schema"])

        return self.main(user_input_query=user_query, table_content=self.table_md)

    def solve_subquery(self, sub_query: str) -> str:
        """
        Solves a natural language subqeury using the SQL exectution.
        """
        sql_result = self.nl2sql(schema_list=self.table_schema, user_query=sub_query)
        generated_sql = self.indexer.last_query
        return self.subquery_summarizer(
            original_content=self.table_md,
            table_schema=self.table_schema,
            gnerated_sql=generated_sql,
            sql_execute_result=sql_result,
            user_query=self.user_query,
        )


if __name__ == "__main__":
    indexer = TableRAGIndexer(
        vdb_config=MilvusVDBConfig.from_local("examples/TableRAG/index2.db"),
        sql_config=SQLiteConfig(db_path="examples/TableRAG/tables.db"),
    )
    agent = TableRAGAgent(config=TableRAGConfig(), indexer=indexer)
    # # x = indexer.sql_query("SELECT * FROM t_5th_new_zealand_parliament_0")
    # # print(x)
    # x = agent(user_query="Who is the New Zealand Parliament Member for Canterbury")
    # print(x)
    agent.push_to_hub("test/test")
