import requests

# Read text file
with open("conversation.txt", "r", encoding="utf-8") as f:
    file_content = f.read()

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
         {"role": "system", "content": "You are an assistant that summarizes a debate and picks the winner in the end."},
        {"role": "user", "content": file_content}  
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": False
}

response = requests.post(url, json=payload)

print(response.json()["choices"][0]["message"]["content"])
