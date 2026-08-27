"""Current-position context for LLM requests: SGF text plus a board screenshot.

Builds the multimodal message parts used by vision-capable chat models: the
game's SGF as a text part, and the board widget rendered to PNG and inlined as
an ``image_url`` data URI. The SGF is built from the line of play leading to
the current node only, so abandoned variations do not confuse the model.
"""

import base64
import os
import tempfile

from katrain.core.constants import OUTPUT_ERROR

SGF_HEADER = "(;GM[1]FF[4]CA[UTF-8]SZ[{size}]"
# X-prefixed properties are private extensions in SGF; XG carries the GTP coordinates
# (e.g. ;B[dd]XG[D16]) so text models do not need the SGF y-flip convention.


def current_position_sgf(katrain):
    """SGF text of the game up to and including the current node.

    Each move node carries an ``XG`` property with the move's GTP coordinates,
    next to the standard SGF value.
    """
    game = katrain.game
    path = []
    node = game.current_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()  # root .. current

    root = path[0]
    size = root.get_property("SZ", 19)
    board_size = (size, size) if isinstance(size, int) else (int(size.split(":")[0]),) * 2
    parts = [SGF_HEADER.format(size=board_size[0])]
    for prop in ("KM", "HA", "RU", "PB", "PW"):
        value = root.get_property(prop, None)
        if value:
            parts.append(f"{prop}[{value}]")
    for n in path[1:]:
        move = n.move
        if move:
            parts.append(f";{n.player}[{move.sgf(board_size)}]")
            if not move.is_pass:  # GTP coordinates of the played point, "pass" adds nothing
                parts.append(f"XG[{move.gtp()}]")
    parts.append(")")
    return "".join(parts)


def board_screenshot_data_uri(katrain):
    """Render the board widget to PNG, return it as a ``data:`` URI string."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="katrain-board-")
    os.close(fd)
    try:
        katrain.board_gui.export_to_png(path)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception as e:
        katrain.log(f"Board screenshot for LLM failed: {e}", OUTPUT_ERROR)
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def position_context_parts(katrain):
    """Multimodal ``content`` parts describing the current position, or ``None``."""
    parts = []
    try:
        sgf = current_position_sgf(katrain)
    except Exception as e:
        katrain.log(f"Building SGF context for LLM failed: {e}", OUTPUT_ERROR)
        sgf = None
    if sgf:
        parts.append({"type": "text", "text": f"Current position SGF:\n{sgf}"})
    image = board_screenshot_data_uri(katrain)
    if image:
        parts.append({"type": "image_url", "image_url": {"url": image}})
    return parts or None
