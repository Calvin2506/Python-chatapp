from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os, time

load_dotenv()

# MongoDB Setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["Chat_history"]
collection = db["conversations"]

def main():
    session_id = input("Enter your username: ").strip()

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0,
        model_kwargs={"streaming": True}
    )

    print(f"\nWelcome {session_id}! I'm your personal AI agent. Type 'quit' to exit.")
    print("Type 'history' to view your previous conversations.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print(f"Chat closed by {session_id}. Have a nice day!", flush=True)
                break

            if user_input.lower() == "history":
                print("\nPrevious Conversations:")
                history = collection.find({"session_id": session_id}).sort("timestamp", 1)
                for convo in history:
                    time_str = convo["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{time_str}]\n{convo['user']}\n{convo['bot']}")
                continue

            print("\nAgent: ", end="", flush=True)
            start = time.time()

            bot_response = ""
            for chunk in model.stream([HumanMessage(content=user_input)]):
                if hasattr(chunk, "content"):
                    print(chunk.content, end="", flush=True)
                    bot_response += chunk.content

            print(f"\n Responded in {round(time.time() - start, 2)} seconds")

            # Save to MongoDB
            collection.insert_one({
                "session_id": session_id,
                "timestamp": datetime.now(),
                "user": user_input,
                "bot": bot_response
            })

        except KeyboardInterrupt:
            print(f"\nChat closed by {session_id}. Have a nice day!", flush=True)
            break

if __name__ == "__main__":
    main()
