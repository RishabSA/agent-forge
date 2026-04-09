from dotenv import load_dotenv
from tui import AgentForgeTUI

load_dotenv()


if __name__ == "__main__":
    app = AgentForgeTUI()
    app.run()
