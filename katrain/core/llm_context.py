"""Current-position context for LLM requests: SGF text plus a board screenshot.

Builds the multimodal message parts used by vision-capable chat models: the
game's SGF as a text part, and the board widget rendered to PNG and inlined as
an ``image_url`` data URI. The SGF is built from the line of play leading to
the current node only, so abandoned variations do not confuse the model.

The notes explaining the SGF to the model are translated through ``i18n``, so
they follow the app's UI language: a Chinese UI sends Chinese notes.
"""

import base64
import os
import tempfile

from katrain.core.constants import OUTPUT_ERROR
from katrain.core.lang import i18n

SGF_HEADER = "(;GM[1]FF[4]CA[UTF-8]SZ[{size}]"
# X-prefixed properties are private extensions in SGF; XG carries the GTP coordinates
# (e.g. ;B[dd]XG[D16]) so text models do not need the SGF y-flip convention.
# Star points are only valid on a 19x19 board; smaller boards have different ones.
STAR_POINTS_19_CORNERS = "D4, Q4, D16, Q16"
STAR_POINTS_19_SIDES = "K4, D10, Q10, K16"
STAR_POINTS_19_CENTER = "K10"  # tengen


def star_points_19_text():
    """The nine star points as one localized string; the GTP points themselves do not translate."""
    return "; ".join(
        [
            i18n._("llm prompt star corners").format(points=STAR_POINTS_19_CORNERS),
            i18n._("llm prompt star sides").format(points=STAR_POINTS_19_SIDES),
            i18n._("llm prompt star center").format(point=STAR_POINTS_19_CENTER),
        ]
    )


def sgf_prompt_text(sgf, board_size_x):
    """SGF plus the notes a text model needs to read it: what XG means, and the star points.

    Looked up at call time, so the notes follow the UI language also after a
    language switch mid-session.
    """
    notes = [i18n._("llm prompt sgf intro"), sgf]
    if board_size_x == 19:
        notes.append(i18n._("llm prompt star points").format(star_points=star_points_19_text()))
    return "\n".join(notes)


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
    board_size_x = 19
    try:
        sgf = current_position_sgf(katrain)
        size = katrain.game.root.get_property("SZ", 19)
        board_size_x = size if isinstance(size, int) else int(str(size).split(":")[0])
    except Exception as e:
        katrain.log(f"Building SGF context for LLM failed: {e}", OUTPUT_ERROR)
        sgf = None
    if sgf:
        parts.append({"type": "text", "text": sgf_prompt_text(sgf, board_size_x)})
    image = board_screenshot_data_uri(katrain)
    if image:
        parts.append({"type": "image_url", "image_url": {"url": image}})
    return parts or None
