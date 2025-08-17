from modaic import PrecompiledAgent, PrecompiledConfig
from modaic.hub import get_user_info
from dotenv import load_dotenv
import os

load_dotenv()

print(get_user_info(os.getenv("MODAIC_TOKEN")))
