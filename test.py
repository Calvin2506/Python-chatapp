from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

print("MONGODB_URI =", os.getenv("MONGODB_URI"))
