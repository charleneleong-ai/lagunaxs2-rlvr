"""Independent test oracle: render a concrete tool call in each scaffold format.

Kept separate from `laguna_rlvr.scaffold.render_instructions` on purpose — it's the round-trip
counterpart that parse_call must recover, so it must not share code with production rendering.
"""


def emit_call(fmt: str, name: str, arg: str, value: str) -> str:
    if fmt == "line":
        return f"{name}: {value}"
    if fmt == "xml":
        return f'<tool_call>{{"name": "{name}", "arguments": {{"{arg}": "{value}"}}}}</tool_call>'
    if fmt == "poolside":
        return f"<tool_call>{name}\n<arg_key>{arg}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>"
    return f'{{"tool": "{name}", "{arg}": "{value}"}}'   # json
