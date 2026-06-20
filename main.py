from dotenv import load_dotenv
from src.groq_llm import GroqLLM

# Load the environment variables from the .env file
load_dotenv()

def test_basic_call():
    print("Initializing Groq LLM...")
    # Instantiate our Groq model (uses the default model we set up)
    llm = GroqLLM()

    # Prepare the message payload
    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Explain Object-Oriented Programming in one short sentence."}
    ]

    print("Sending request to Groq...")
    # Call our generic method
    response = llm.generate_response(messages=messages)

    # Extract and print the actual text content from the response
    reply = response.choices[0].message.content
    print("\n--- LLM Response ---")
    print(reply)
    print("--------------------\n")

if __name__ == "__main__":
    test_basic_call()