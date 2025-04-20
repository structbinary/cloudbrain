from colorama import Fore, Style
from enum import Enum


class AgentColor(Enum):
    RETRIEVER = Fore.LIGHTBLUE_EX
    GENERATOR = Fore.YELLOW
    WEBSEARCH = Fore.LIGHTGREEN_EX
    PUBLISHER = Fore.MAGENTA
    REVIEWER = Fore.CYAN
    REVISOR = Fore.LIGHTWHITE_EX
    MASTER = Fore.LIGHTYELLOW_EX
    ROUTER = Fore.LIGHTRED_EX
    SEARCHER = Fore.LIGHTGREEN_EX


def print_agent_output(output:str, agent: str="GENERATOR"):
    print(f"{AgentColor[agent].value}{agent}: {output}{Style.RESET_ALL}")