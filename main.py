from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates
import requests
import os
import json
import re

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

PROMPT_TEMPLATE = """
You are an AWS access management assistant. Given the following user request, provide the exact AWS CLI command that an admin should run to grant the necessary access. 
Also, provide any additional notes if necessary.

    AWS Account ID: {account_id}

    User IAM ID: {user_arn}

    AWS Region: {region}

    User Request: {request}

    Format your response as follows:

    **Command:**

    ```bash
    <aws cli command>
    ```

    **Notes:**
    <any additional notes>"""

# Pydantic model for the POST /tickets request body
class TicketRequest(BaseModel):
    llm: str
    model: str
    request: str
    user_arn: str  # Added user_arn to the request body

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Tickets endpoint to process user requests
@app.post("/tickets")
def create_ticket(ticket: TicketRequest):
    # Retrieve environment variables
    aws_account_id = os.getenv("AWS_ACCOUNT_ID")
    aws_region = os.getenv("AWS_REGION")
    
    # Check if required environment variables are set
    if not all([aws_account_id, aws_region]):
        raise HTTPException(status_code=500, detail="Missing required environment variables")
    
    # Construct the prompt using the configurable template
    prompt = PROMPT_TEMPLATE.format(
        account_id=aws_account_id,
        user_arn=ticket.user_arn,  # Use user_arn from request body
        region=aws_region,
        request=ticket.request
    )
    
    # Call the appropriate LLM based on the 'llm' field
    if ticket.llm == "openai":
        response = call_openai(ticket.model, prompt)
    elif ticket.llm == "ollama":
        response = call_ollama(ticket.model, prompt)
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM specified. Use 'openai' or 'ollama'")
    
    # Parse the LLM response
    command, notes, tokens = parse_llm_response(response, ticket.llm)
    
    # Return the JSON response
    return {
        "model": ticket.model,
        "llm": ticket.llm,
        "tokens": tokens,
        "command": command,
        "notes": notes
    }

# Function to call OpenAI API
def call_openai(model: str, prompt: str):
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_token = os.getenv("OPENAI_TOKEN")
    if not openai_token:
        raise HTTPException(status_code=500, detail="OpenAI token not configured")
    
    url = f"{openai_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_token}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# Function to call Ollama API (assumed API structure)
def call_ollama(model: str, prompt: str):
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    if not ollama_base_url:
        raise HTTPException(status_code=500, detail="Ollama base URL not configured")
    
    url = f"{ollama_base_url}/api/generate"
    print(url)
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "system": "You are AWS clie expert assistant",
        "prompt": prompt,
    }
    try:
        # Use streaming to handle multiple JSON objects
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        # Accumulate the full response content
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Decode each line and extract the 'response' field
                chunk = line.decode('utf-8')
                try:
                    json_chunk = json.loads(chunk)
                    if "response" in json_chunk:
                        full_response += json_chunk["response"]
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        
        # Return a mock JSON response compatible with parse_llm_response
        print(full_response)
        return {"response": full_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {str(e)}")

# Function to parse LLM response
def parse_llm_response(response, llm: str):
    if llm == "openai":
        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("total_tokens")
    elif llm == "ollama":
        content = response.get("response")  # Assumed key; adjust based on actual Ollama API
        tokens = response.get("tokens")    # May not be available
    else:
        raise ValueError("Invalid LLM")
    
    # Extract command and notes using regex
    command_match = re.search(r'\*\*Command:\*\*\s*```bash\s*(.*?)\s*```', content, re.DOTALL)
    notes_match = re.search(r'\*\*Notes:\*\*\s*(.*)', content, re.DOTALL)
    
    command = command_match.group(1).strip() if command_match else "Command not found"
    notes = notes_match.group(1).strip() if notes_match else "No notes provided"
    
    return command, notes, tokens

# Run the app with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086, reload=True)