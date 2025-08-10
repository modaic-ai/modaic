# Create A Custom Context Class

In this guide we will show you how to create your own context class. The context engineering toolkit in Modaic is extremely powerful and flexible. To learn more about how it works, check out the [context engineering guide](./context_engineering.md). In this guide we will show you how to create a custom user profile and organization profile context class for a social networking agent.

## The UserProfile class
First we will define the `UserProfile` class as well as its `SerializedUserProfile` class.
```python
from modaic.context import Atomic, Molecular, SerializedContext
import requests
from PIL import Image
from io import BytesIO

# First we define UserProfile's SerializedContext class.
# As you can see below, only name, age, email,and description will be serialized. profile_pic will be loaded during construction.
class SerializedUserProfile(SerializedContext):
    name: str
    age: int
    description: str
    email: str

class UserProfile(Atomic):
    serialized_schema = SerializedUserProfile # !!! Super important for serialization and deserialization.
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
    # We don't explicitly need to do this since by default the readme method will return self.serialized_schema
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

1. We defined the `UserProfile` class that extends from the `Atomic` context type. It has the attributes `name`, `age`, `description`, `email`, and `profile_pic`. Profile pic is dynamically loaded from the backend.

2. We defined the `SerializedUserProfile` which determines serialization behavior for the `UserProfile` class. It expects the attributes `name`, `age`, `description`, and `email`. It will ignore the `profile_pic` attribute.

3. We implemented the `embedme` method which returns the `description` of the user.

4. We implemented the `readme` method which returns a string that represents the user profile.

Let's see what this looks like in action!

```python
from modaic.databases import VectorDatabase, MilvusVDBConfig
from modaic.utils import Embedder
from modaic.context import Source, SourceType
import dspy

milvus_config = MilvusVDBConfig.from_local(file_path="index.db")
vector_db = VectorDatabase(
    config=milvus_config,
    embedder=Embedder(model="openai/text-embedding-3-small"),
)

user_profile1 = UserProfile(
    name="John Doe",
    age=30,
    description="John Doe is a software engineer at Google. He also loves dogs.",
    email="john.doe@gmail.com",
    source=Source(origin="https://example.com/john_doe", type=SourceType.URL),
)
user_profile2 = UserProfile(
    name="Jane Smith",
    age=25,
    description="Jane Smith is a software engineer at Meta.",
    email="jane.smith@gmail.com",
    source=Source(origin="https://example.com/jane_smith", type=SourceType.URL),
)
vector_db.create_collection("user_profiles", payload_schema=SerializedUserProfile)

# Add records to the vector database. The `add_records` method will automatically call the `.embedme()` function and pass the result into the Embedder to embed the context.
vector_db.add_records(
    "user_profiles",
    [user_profile1, user_profile2],
)

# Now lets search for user profiles.
meta_employee = vector_db.search(
    "user_profiles",
    "Someone who works at Meta",
    k=1,
)
# This should *hopefully* return Jane Smith.

# Now lets have an give us a summary of the meta employee.

summarizer = dspy.Predict("user_profile -> summary")
summarizer.set_lm(dspy.LM(model="openai/gpt-4o-mini"))

summary = summarizer(user_profile=meta_employee.readme())
```
What happened here?

1. We created two user profiles and added them to the vector database. The vector database automatically used `description` to embed the context. Since that is what `embedme()` returns

2. We fetched the `meta_employee` context from the vector database, then summarized the profile by feeding in the result of the `readme` method to a dspy.Predict module. (just a simple LLM call)

### Alternate Implementation
What if instead we wanted to embed users based on what they look like? All we would have to do is change the `embedme` method to return the profile picture.

```python
def embedme(self) -> PIL.Image.Image:
    return self.profile_pic
```

## The OrganizationProfile class
Now that we have a `UserProfile` class, we can create an `OrganizationProfile` class. That is composed of multiple `UserProfile`s. Since UserProfile can be chunked, it is a good idea to make it extend the `Molecular` context type.

```python
from modaic.context import Molecular
import requests

class SerializedOrganizationProfile(SerializedContext):
    name: str
    website_url: str

class OrganizationProfile(Molecular):
    def __init__(self, name: str, website_url: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.website_url = website_url
    # We won't define a proper embedme since we aren't going to be embedding organization profiles.
    def embedme(self) -> str: 
        pass
    
    def readme(self) -> str:
        return f"""
        Organization Name: {self.name}
        Description: {self.description}
        """
    
    # We will also define a method that fetches an organization profile from a url.
    @staticmethod
    def from_url(url: str) -> OrganizationProfile:
        org = requests.get(url).json().data
        return OrganizationProfile(**org)

def chunk_by_user(org_profile: OrganizationProfile) -> list[UserProfile]:
    users = requests.get(f'{org_profile.source.origin}/users').json().data
    return [UserProfile(**user) for user in users]

# Now lets create an organization profile.
org_profile1 = OrganizationProfile(
    name="Google",
    description="Google is a search engine company.",
    source=Source(origin="https://api.example.com/google", type=SourceType.URL),
)
org_profile2 = OrganizationProfile(
    name="Meta",
    description="Meta is a social networking company.",
    source=Source(origin="https://api.example.com/meta", type=SourceType.URL),
)

vector_db.create_collection("user_profiles", payload_schema=SerializedOrganizationProfile)

org_profile1.chunk_with(chunk_fn=chunk_by_user)
org_profile2.chunk_with(chunk_fn=chunk_by_user)

vector_db.add_records(
    "user_profiles",
    org_profile1.chunks + org_profile2.chunks,
)

# Now lets search for user profiles.
dog_lover = vector_db.search(
    "user_profiles",
    "Someone who loves dogs",
    k=1,
)

# We can use the source attribute to get the organization this profile belongs to.
org_url = dog_lover.source.origin
org_profile = OrganizationProfile.from_url(org_url)
```

## Advanced Usage: Grabbing values at serialization time
Lets say we want to add a last accessed timestamp to out user profile for filtering in the vector database.
```python
import datetime

class SerializedUserProfile(SerializedContext):
    name: str
    age: int
    description: str
    email: str
    last_accessed: str

class UserProfile(Molecular):
    def __init__(self, name: str, age: int, description: str, email: str, last_accessed: datetime.datetime, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.age = age
        self.description = description
        self.email = email
        self.last_accessed = datetime.datetime.now().isoformat()
```
We have a problem! `last_accessed` will refer to the time of constructution, not of serialization. The solution? We can actually add the output of functions to `SerializedUserProfile` as long as they can be called with no arguments.
```python
class SerializedUserProfile(SerializedContext):
    name: str
    age: int
    description: str
    email: str
    last_accessed: str

class UserProfile(Molecular):
    def __init__(self, name: str, age: int, description: str, email: str, last_accessed: datetime.datetime, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.age = age
        self.description = description
        self.email = email
    
    def last_accessed(self) -> str:
        return datetime.datetime.now().isoformat()
```

the `serialize` funtion will automatically look for a function named `last_accessed` and call it to get the value of `last_accessed` during serialization.



