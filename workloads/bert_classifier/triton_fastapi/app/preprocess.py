from typing import List


def validate_message(message: str) -> str:
    return message.strip()


def validate_messages(messages: List[str]) -> List[str]:
    return [message.strip() for message in messages]
