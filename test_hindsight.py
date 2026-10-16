import os
from dotenv import load_dotenv
from hindsight_client import Hindsight

load_dotenv()

client = Hindsight(
    base_url=os.getenv("HINDSIGHT_API_URL"),
    api_key=os.getenv("HINDSIGHT_API_KEY"),
)

BANK_ID = "deal-intelligence"

client.retain(
    bank_id=BANK_ID,
    content="Acme Corp said pricing is their main concern during the sales call."
)

result = client.recall(
    bank_id=BANK_ID,
    query="What is Acme Corp concerned about?"
)

print(result)