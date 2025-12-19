import os

from dotenv import load_dotenv

from modaic.hub import get_user_info

load_dotenv()

print(get_user_info(os.getenv("MODAIC_TOKEN")))
