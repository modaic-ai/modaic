# Welcome to the Modaic Docs

## Getting Started

Install Modaic

```bash
uv add modaic
```
or

```bash
pip install modaic
```


## Modaic Principles

In Modaic there are two types of context. `Molecular` and `Atomic`. Atomic context is the finest granularity of context and is not chunkable. Molecular context is larger pieces of context that can be chunked into smaller `Molecular` or `Atomic` context objects.

## Create a Simple RAG Framework

Lets create a simple agent that can answer questions about the weather.

```python
from modaic import PrecompiledAgent, PrecompiledConfig
import dspy

class WeatherConfig(PrecompiledConfig):
    agent_type = "WeatherAgent" # !! This is super important so you can load the agent later!!

class WeatherAgent(PrecompiledAgent):
    config_class = WeatherConfig # !! This is super important to link the agent to the config!!
    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.summarize = dspy.ReAct(signature="question->answer", tools=[self.get_weather])

    def forward(self, query: str) -> str:
        return self.summarize(query)

    def get_weather(self, city: str) -> str:
        """
        Get the weather in a city.
        """
        return f"The weather in {city} is sunny."

agent = WeatherAgent(PrecompiledConfig())
agent("What is the weather in Tokyo?")
print(response)
```
Response:
```python
Prediction(
    trajectory={'thought_0': 'I need to get the current weather information for Tokyo to answer the question.', 'tool_name_0': 'get_weather', 'tool_args_0': {'city': 'Tokyo'}, 'observation_0': 'The weather in Tokyo is sunny.', 'thought_1': 'I have obtained the weather information for Tokyo, which is sunny. Now I can finalize my response to the question.', 'tool_name_1': 'finish', 'tool_args_1': {}, 'observation_1': 'Completed.'},
    reasoning='The current weather information for Tokyo has been obtained, indicating that it is sunny. This directly answers the question about the weather in Tokyo.',
    answer='The weather in Tokyo is sunny.'
)

```
```python
# Push agent to the hub (optional)
agent.push_to_hub("modaic/weather-agent")
```

## Using the Context Engineering Toolkit
Here we define an Indexer that ingests txt or md files, chunks them, then adds them to a vector database.
```python
import modaic
from modaic.context import LongText, Text
from modaic.databases import MilvusVDBConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MyIndexer(modaic.Indexer):
    def __init__(
        self, vdb_config: MilvusVDBConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedder = modaic.Embedder(model="openai/text-embedding-3-small")
        self.vector_database = VectorDatabase(
            config=vdb_config,
            embedder=self.embedder,
        )
        self.sql_db = SQLDatabase(config=sql_config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.vector_database.create_collection(
            "docs", Text.schema, if_exists="replace"
        )

    def ingest(self, files: List[str]):
        records = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            text_document = LongText(text=text)
            text_document.chunk_text(self.text_splitter.split_text)
            records.extend(text_document.get_chunks())
        self.vector_database.add_records("docs", records)
    def retrieve(self, query: str, k: int = 10) -> List[Text]:
        return self.vector_database.retrieve(query, k)
```

## Define your own Context Class
You can also define your own context in modaic. Here we define a `UserProfile` context class that is an `Atomic` context. Which means it is not chunkable.

```python
from modaic.context import Atomic, Molecular, SerializedContext
import requests
from PIL import Image
from io import BytesIO

# First we define UserProfile's SerializedContext class.
# As you can see below, only name, age, email,and description will be serialized. profile_pic will only be loaded during construction.
class SerializedUserProfile(SerializedContext):
    name: str
    age: int
    description: str
    email: str

class UserProfile(Atomic):
    schema = SerializedUserProfile # !!! Super important for serialization and deserialization.
    def __init__(self, name: str, age: int, description: str, email: str, profile_pic: PIL.Image.Image, **kwargs):
        # All attibutes that will be serialized must match fields of SerializedUserProfile
        super().__init__(**kwargs) # !!! Important. Allows the parent class to initalize source and metadata.
        self.name = name 
        self.age = age
        self.description = description
        self.email = email
        self.profile_pic = self.get_profile_pic()
    
    def get_profile_pic(self) -> PIL.Image.Image:
        response = requests.get(self.source.origin)
        data = response.json()
        img_response = requests.get(data["profile_pic"])
        return Image.open(BytesIO(img_response.content))
    
    # Define the abstract method embedme
    def embedme(self) -> str:
        return self.description
    
    # Define the readme method.
    # We don't explicitly need to do this since by default the readme method will return self.serialize()
    # However, its useful to override when you need custom behavior.
    def readme(self) -> str:
        return f"""
        User Name: {self.name}
        Age: {self.age}
        Email: {self.email}
        Description: {self.description}
        """
```
So what did we do here?

1. We defined the UserProfile class that extends from the Atomic context type. It has the attributes name, age, description, email, and profile_pic. Profile pic is dynamically loaded from the backend.

2. We defined the SerializedUserProfile which determines serialization behavior for the UserProfile class. It expects the attributes name, age, description, and email. It will ignore the profile_pic attribute.

3. We implemented the embedme method which returns the description of the user.

4. We implemented the readme method which returns a string that represents the user profile.

## Bringing it all together 
Lets define a networking agent that emails users you may be interested in meeting.

First lets define an indexer for the user profiles.

```python
from modaic.context import LongText, Text
from modaic.databases import MilvusVDBConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

class UserProfileIndexer(Indexer):
    def __init__(
        self, vdb_config: MilvusVDBConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedder = modaic.Embedder(model="openai/text-embedding-3-small")
        self.vector_database = VectorDatabase(
            config=vdb_config,
            embedder=self.embedder,
        )
        self.vector_database.create_collection(
            "user_profiles", UserProfile.schema, if_exists="append"
        )
    def ingest(self, user_profiles: List[dict]):
        records = []
        for user_profile in user_profiles:
            user_profile = UserProfile(**user_profile)
            records.append(user_profile)
        self.vector_database.add_records("user_profiles", records)
    
    def retrieve(self, query: str, k: int = 10) -> List[SerializedUserProfile]:
        return self.vector_database.retrieve(query, k)
```

Now lets define the NetworkingAgent.

```python
from modaic import PrecompiledAgent, PrecompiledConfig
import dspy

class NetworkingAgentConfig(PrecompiledConfig):
    agent_type = "NetworkingAgent" # !! This is super important so you can load the agent later!!
    milvus_config: MilvusVDBConfig


class NetworkingAgent(PrecompiledAgent):
    config_class = NetworkingAgentConfig # !! This is super important to link the agent to the config!!
    def __init__(self, config: NetworkingAgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.networker = dspy.ReAct(signature="question->answer", tools=[self.send_email, self.get_user_profiles])
        self.indexer = UserProfileIndexer(config.milvus_config)

    def forward(self, query: str) -> str:
        return self.networker(query)

    def send_email(self, email: str, message: str) -> str:
        """
        Send an email to a user.
        """
        # The Doc string above will describe the tool to the ReAct agent.
        return f"Email sent to {email} with message: {message}"
    
    def get_user_profiles(self, query: str) -> List[SerializedUserProfile]:
        """
        Gets user profiles that match the query.
        """
        return self.indexer.retrieve(query, k=10)
    
    def ingest_user_profiles(self, user_profiles: List[dict]):
        """
        Ingests user profiles into the indexer.
        """
        self.indexer.ingest(user_profiles)

config = NetworkingAgentConfig(milvus_config=MilvusVDBConfig.from_local("index.db"))
agent = NetworkingAgent(config)
```
We'll add some user profiles to the indexer.

```python
user_profiles = [
    {
        "name": "John Doe",
        "age": 30,
        "description": "John is a designer",
        "email": "john.doe@example.com",
    },
    {
        "name": "Jane Doe",
        "age": 25,
        "description": "Jane is a software engineer",
        "email": "jane.doe@example.com",
    },
]
agent.ingest_user_profiles(user_profiles)
```
Now lets test it out.
```python
response = agent(question="Which user is a software engineer like me?")
print(response)

```

Response:
```python
Prediction(
    trajectory={'thought_0': 'I need to find user profiles that match the description of being a software engineer. This will help identify users who share a similar profession.', 'tool_name_0': 'get_user_profiles', 'tool_args_0': {'description': 'software engineer'}, 'observation_0': [UserProfile(name='John Doe', age=30, description='John is a designer', email='john.doe@example.com'), UserProfile(name='Jane Doe', age=25, description='Jane is a software engineer', email='jane.doe@example.com')], 'thought_1': 'I found one user, Jane Doe, who is a software engineer. I can now finish the task since I have the information needed to identify a user with the same profession.', 'tool_name_1': 'finish', 'tool_args_1': {}, 'observation_1': 'Completed.'},
    reasoning='I identified a user who shares the same profession as a software engineer. The user is Jane Doe, who is explicitly described as a software engineer. This matches the criteria I was looking for.',
    response='The user who is a software engineer like you is Jane Doe.'
)
```
Lets test out the emailing functionality.
```python
response = agent(
    question="Can you email a designer in my network? I need someone to design the UI for my website."
)
print(response)

```

Response:
```python
Prediction(
    trajectory={'thought_0': "I need to find a designer in the user's network before I can email them. I will use the get_user_profiles tool to retrieve user profiles that might include designers.", 'tool_name_0': 'get_user_profiles', 'tool_args_0': {'description': 'designer'}, 'observation_0': [UserProfile(name='John Doe', age=30, description='John is a designer', email='john.doe@example.com'), UserProfile(name='Jane Doe', age=25, description='Jane is a software engineer', email='jane.doe@example.com')], 'thought_1': "I found a designer named John Doe in the user's network. I will proceed to email him regarding the UI design for the website.", 'tool_name_1': 'email_user', 'tool_args_1': {'email': 'john.doe@example.com', 'message': "Hi John, I hope you're doing well! I need someone to design the UI for my website and I thought of you. Would you be interested in discussing this project further?"}, 'observation_1': "Email sent to john.doe@example.com with message: Hi John, I hope you're doing well! I need someone to design the UI for my website and I thought of you. Would you be interested in discussing this project further?", 'thought_2': 'I have successfully emailed John Doe about the UI design for the website. Since the task is complete, I will finish the process.', 'tool_name_2': 'finish', 'tool_args_2': {}, 'observation_2': 'Completed.'},
    reasoning="I identified a designer named John Doe in the user's network and successfully emailed him regarding the UI design for the website. The email was sent with a clear message expressing the user's need for a designer and inviting John to discuss the project further.",
    response='I have emailed John Doe about the UI design for your website. He should get back to you soon.'
) 

```

Push to the hub (optional)
```python
agent.push_to_hub("modaic/networking-agent")
```

