from dotenv import load_dotenv
from tui import MultiAgentTUI

load_dotenv()


if __name__ == "__main__":
    app = MultiAgentTUI()
    app.run()
