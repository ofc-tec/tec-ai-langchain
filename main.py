
import os
from dotenv import load_dotenv


load_dotenv()

def main():
    print("Hello from tec-agentic-ai!")
    print(f' OPEN AI KEY{os.getenv("OPENAI_API_KEY")}')

if __name__ == "__main__":
    main()


