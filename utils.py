"""
Utility functions for formatting and displaying conversation messages.

This module provides functions to render message objects in a structured
and visually appealing way in the console using the `rich` library.
"""

# Import necessary standard libraries
import json
# Import typing hints for clear function signatures
from typing import Any, List

# Import components from the rich library for enhanced console output
from rich.console import Console
from rich.panel import Panel

# Initialize a global Console object from the rich library.
# This object will be used for all styled output to the terminal.
console = Console()


def format_message_content(message: Any) -> str:
    """
    Converts the content of a message object into a displayable string.

    This function handles simple string content as well as complex list-based
    content, such as tool calls, by parsing and formatting them appropriately.

    Args:
        message: A message object that has a 'content' attribute.

    Returns:
        A formatted string representation of the message content.
    """
    # Retrieve the content from the message object.
    content = message.content

    # Check if the content is a simple string.
    if isinstance(content, str):
        # If it is, return it directly.
        return content
    # Check if the content is a list, which often indicates complex data like tool calls.
    elif isinstance(content, list):
        # Initialize an empty list to hold formatted parts of the content.
        parts = []
        # Iterate over each item in the content list.
        for item in content:
            # If the item is a simple text block.
            if item.get("type") == "text":
                # Append the text directly to our parts list.
                parts.append(item["text"])
            # If the item represents a tool being used.
            elif item.get("type") == "tool_use":
                # Format a string to indicate a tool call, including the tool's name.
                tool_call_str = f"\n🔧 Tool Call: {item.get('name')}"
                # Format the tool's input arguments as a pretty-printed JSON string.
                tool_args_str = f"   Args: {json.dumps(item.get('input', {}), indent=2)}"
                # Add the formatted tool call strings to our parts list.
                parts.extend([tool_call_str, tool_args_str])
        # Join all the formatted parts into a single string, separated by newlines.
        return "\n".join(parts)
    # For any other type of content.
    else:
        # Convert the content to a string as a fallback.
        return str(content)


def format_messages(messages: List[Any]) -> None:
    """
    Formats and displays a list of messages using rich Panels.

    Each message is rendered inside a styled panel, with a title and border
    color that corresponds to its role (e.g., Human, AI, Tool).

    Args:
        messages: A list of message objects to be displayed.
    """
    # Iterate through each message object in the provided list.
    for m in messages:
        # Determine the message type by getting the class name and removing "Message".
        msg_type = m.__class__.__name__.replace("Message", "")
        # Get the formatted string content of the message using our helper function.
        content = format_message_content(m)

        # Define default arguments for the rich Panel.
        panel_args = {"title": f"📝 {msg_type}", "border_style": "white"}

        # Customize panel appearance based on the message type.
        # If the message is from a human user.
        if msg_type == "Human":
            # Update the title and set the border color to blue.
            panel_args.update(title="🧑 Human", border_style="blue")
        # If the message is from the AI assistant.
        elif msg_type == "Ai":
            # Update the title and set the border color to green.
            panel_args.update(title="🤖 Assistant", border_style="green")
        # If the message is a tool's output.
        elif msg_type == "Tool":
            # Update the title and set the border color to yellow.
            panel_args.update(title="🔧 Tool Output", border_style="yellow")

        # Create a Panel with the formatted content and customized arguments.
        # Then, print the panel to the console.
        console.print(Panel(content, **panel_args))


def format_message(messages: List[Any]) -> None:
    """
    Alias for the format_messages function.

    This provides backward compatibility for any code that might still
    use the singular name `format_message`.

    Args:
        messages: A list of message objects to be displayed.
    """
    # Call the main format_messages function to perform the rendering.
    format_messages(messages)