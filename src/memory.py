import json
import os
from typing import List, Dict, Any
from datetime import datetime


class ConversationMemory:
    """
    Manages conversation history for the Mama Amaka agent.
    Stores and retrieves conversations to provide context for future interactions.
    """

    def __init__(
        self, memory_file: str = "conversation_history.json", max_history: int = 10
    ):
        """
        Initialize the conversation memory.

        Args:
            memory_file: File path to store conversation history
            max_history: Maximum number of conversation turns to remember (context window)
        """
        self.memory_file = memory_file
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load existing conversation history from file
        self.load_from_file()

    def add_message(
        self, role: str, content: str, metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Either "user" or "assistant"
            content: The message content
            metadata: Optional metadata (sources, scores, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.conversation_history.append(message)

        # Keep only the last max_history messages to manage context window
        if (
            len(self.conversation_history) > self.max_history * 2
        ):  # *2 because each turn is user + assistant
            self.conversation_history = self.conversation_history[
                -(self.max_history * 2) :
            ]

        # Save to file after each message
        self.save_to_file()

    def get_context(self) -> str:
        """
        Get formatted conversation history for use as context in prompts.
        Returns only the recent conversation turns, excluding metadata.

        Returns:
            Formatted string of conversation history
        """
        if not self.conversation_history:
            return ""

        context_lines = ["--- Recent Conversation History ---"]

        # Get last few exchanges (max_history turns)
        recent_messages = self.conversation_history[-(self.max_history * 2) :]

        for msg in recent_messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_lines.append(
                f"{role}: {content[:100]}..."
                if len(content) > 100
                else f"{role}: {content}"
            )

        context_lines.append("--- End of History ---\n")
        return "\n".join(context_lines)

    def get_last_n_exchanges(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get the last n user-assistant exchanges.

        Args:
            n: Number of exchanges to return

        Returns:
            List of dictionaries with 'user' and 'assistant' keys
        """
        exchanges = []
        messages = self.conversation_history

        i = len(messages) - 1
        while i >= 0 and len(exchanges) < n:
            if (
                messages[i]["role"] == "assistant"
                and i > 0
                and messages[i - 1]["role"] == "user"
            ):
                exchanges.insert(
                    0,
                    {
                        "user": messages[i - 1]["content"],
                        "assistant": messages[i]["content"],
                    },
                )
                i -= 2
            else:
                i -= 1

        return exchanges

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        # Remove the file if it exists
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)

    def save_to_file(self) -> None:
        """Save conversation history to a JSON file."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "created_at": datetime.now().isoformat(),
                        "history": self.conversation_history,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def load_from_file(self) -> None:
        """Load conversation history from a JSON file."""
        if not os.path.exists(self.memory_file):
            self.conversation_history = []
            return

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.conversation_history = data.get("history", [])
                self.session_id = data.get("session_id", self.session_id)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            self.conversation_history = []

    def get_summary(self) -> str:
        """
        Get a brief summary of the conversation.

        Returns:
            Summary string with number of messages and key topics
        """
        if not self.conversation_history:
            return "No conversation history."

        user_msgs = [m for m in self.conversation_history if m["role"] == "user"]
        assistant_msgs = [
            m for m in self.conversation_history if m["role"] == "assistant"
        ]

        return f"Conversation Summary: {len(user_msgs)} user messages, {len(assistant_msgs)} assistant responses. Session ID: {self.session_id}"

    def export_conversation(self, filepath: str = None) -> str:
        """
        Export the conversation to a text file.

        Args:
            filepath: Path to export to. If None, uses default name with timestamp.

        Returns:
            Path to the exported file
        """
        if filepath is None:
            filepath = f"conversation_export_{self.session_id}.txt"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Mama Amaka Conversation Export\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")

                for msg in self.conversation_history:
                    role = msg["role"].upper()
                    content = msg["content"]
                    timestamp = msg.get("timestamp", "")
                    f.write(f"[{timestamp}] {role}:\n{content}\n\n")

            return filepath
        except Exception as e:
            print(f"Error exporting conversation: {e}")
            return None
