import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://api.hindsight.vectorize.io/v1/default/banks/deal-intelligence/memories"

headers = {
    "Authorization": f"Bearer {os.getenv('HINDSIGHT_API_KEY')}"
}

response = requests.delete(url, headers=headers)

print(response.status_code)
print(response.text)