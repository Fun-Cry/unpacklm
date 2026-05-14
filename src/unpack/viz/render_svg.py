"""
unpack.viz.render_svg - Circuit path visualization.

All components branch LEFT of the residual stream.
Streams only extend up to the highest active layer per position.
Target token shown at the top.
"""

from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from unpack.viz.graph import CircuitGraph, Hop, VisPath


def render_svg(graph: "CircuitGraph", width=900, layer_height=56,
               pos_spacing=160, margin=80, show_all_layers=False) -> str:

    positions = graph.active_positions
    if not positions:
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80">'
                '<text x="20" y="40" fill="#999">No paths</text></svg>')

    num_layers = graph.num_layers
    n_cols = len(positions)
    pos_to_col = {pos: i for i, pos in enumerate(positions)}

    total_rows = num_layers + 2
    height = total_rows * layer_height + margin * 2 + 50
    total_width = max(width, n_cols * pos_spacing + margin * 2)

    def y_of(layer: float) -> float:
        row_from_bottom = layer + 1
        row_from_top = num_layers - row_from_bottom
        return margin + 45 + row_from_top * layer_height

    def x_of(col: int) -> float:
        span = (n_cols - 1) * pos_spacing
        start = (total_width - span) / 2
        return start + col * pos_spacing

    # ── Component layout: everything LEFT of stream ──
    box_w, box_h = 58, 20
    branch = box_w / 2 + 18
    detour_h = 12

    comp_info = {}
    # Track highest layer per position for stream truncation
    pos_max_layer = defaultdict(lambda: -1)
    pos_min_layer = defaultdict(lambda: num_layers)

    for vp in graph.paths:
        for h in vp.hops:
            key = (h.component, h.position)
            if key in comp_info or h.position not in pos_to_col:
                continue
            col = pos_to_col[h.position]
            stream_x = x_of(col)
            cy = y_of(h.layer)

            if h.is_embedding:
                cx = stream_x
                side = "center"
            else:
                cx = stream_x - branch
                side = "left"

            in_y = cy + detour_h
            out_y = cy - detour_h

            comp_info[key] = {
                "cx": cx, "cy": cy,
                "stream_x": stream_x,
                "in_y": in_y, "out_y": out_y,
                "side": side,
            }

            layer_int = int(h.layer) if not h.is_embedding else -1
            if layer_int > pos_max_layer[h.position]:
                pos_max_layer[h.position] = layer_int
            if layer_int < pos_min_layer[h.position]:
                pos_min_layer[h.position] = layer_int

    # ── SVG ──
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'width="{total_width}" height="{height}">')

    # Defs
    colors_used = {vp.color or "#c0392b" for vp in graph.paths}
    svg.append('<defs>')
    svg.append('<filter id="sh" x="-4%" y="-4%" width="108%" height="108%">'
               '<feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-opacity="0.06"/></filter>')
    for c in colors_used:
        s = _css_safe(c)
        svg.append(f'<marker id="a-{s}" viewBox="0 0 8 8" refX="4" refY="4" '
                   f'markerWidth="4" markerHeight="4" orient="auto">'
                   f'<path d="M0,1 L8,4 L0,7Z" fill="{c}" opacity="0.8"/></marker>')
    svg.append('</defs>')

    # Background
    svg.append(f'<rect width="{total_width}" height="{height}" fill="#fdfdfd"/>')

    # ── Residual streams (truncated per position) ──
    for pos in positions:
        col = pos_to_col[pos]
        cx = x_of(col)
        max_layer = pos_max_layer.get(pos, -1)
        min_layer = pos_min_layer.get(pos, -1)

        if max_layer < 0:
            continue

        stream_top = y_of(max_layer) - detour_h - 8
        stream_bot = y_of(min_layer) + detour_h + 8
        if min_layer == -1:
            stream_bot = y_of(-1) + box_h / 2

        svg.append(f'<line x1="{cx}" y1="{stream_top}" x2="{cx}" y2="{stream_bot}" '
                   f'stroke="#e0e0e0" stroke-width="3.5" stroke-linecap="round"/>')
        # Upward arrow at top
        svg.append(f'<polygon points="{cx-4},{stream_top+6} {cx},{stream_top-2} '
                   f'{cx+4},{stream_top+6}" fill="#d0d0d0"/>')

    # ── Target token label at top of last-position stream ──
    if graph.tokens and positions:
        # Find the position with the highest layer (where output goes)
        target_pos = max(positions, key=lambda p: pos_max_layer.get(p, -1))
        target_col = pos_to_col[target_pos]
        target_x = x_of(target_col)
        target_top = y_of(pos_max_layer[target_pos]) - detour_h - 22
        
        # Show target token if available
        target_label = "target"
        if graph.target_token:
            target_label = graph.target_token
        svg.append(f'<text x="{target_x}" y="{target_top}" text-anchor="middle" '
                   f'fill="#888" font-family="Helvetica,sans-serif" '
                   f'font-size="10" font-style="italic">\u2191 {_esc(target_label)}</text>')

    # ── Layer labels (only for layers that have components) ──
    active_layers = set()
    for vp in graph.paths:
        for h in vp.hops:
            if not h.is_embedding:
                active_layers.add(int(h.layer))

    for L in sorted(active_layers):
        ly = y_of(L)
        svg.append(f'<text x="{margin - 40}" y="{ly + 4}" text-anchor="end" '
                   f'fill="#c8c8c8" font-family="Helvetica,sans-serif" font-size="9">L{L}</text>')
    if any(h.is_embedding for vp in graph.paths for h in vp.hops):
        ey = y_of(-1)
        svg.append(f'<text x="{margin - 40}" y="{ey + 4}" text-anchor="end" '
                   f'fill="#c8c8c8" font-family="Helvetica,sans-serif" font-size="9">emb</text>')

    # ── Token labels at bottom ──
    for pos in positions:
        cx = x_of(pos_to_col[pos])
        ty = height - 28
        tok = ""
        if graph.tokens and 0 <= pos < len(graph.tokens):
            tok = graph.tokens[pos].replace("\u0120", " ").strip()
            if len(tok) > 12: tok = tok[:11] + "\u2026"
        svg.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" fill="#444" '
                   f'font-family="Helvetica,sans-serif" font-size="11" '
                   f'font-weight="600">{_esc(tok)}</text>')
        svg.append(f'<text x="{cx}" y="{ty + 14}" text-anchor="middle" fill="#aaa" '
                   f'font-family="Helvetica,sans-serif" font-size="9">pos {pos}</text>')

    for i in range(len(positions) - 1):
        if positions[i+1] - positions[i] > 1:
            mx = (x_of(i) + x_of(i+1)) / 2
            svg.append(f'<text x="{mx}" y="{height - 28}" text-anchor="middle" '
                       f'fill="#d0d0d0" font-size="12">\u22ef</text>')

    # ── Component boxes with read/write stubs ──
    for (comp, pos), info in comp_info.items():
        cx, cy = info["cx"], info["cy"]
        sx = info["stream_x"]
        in_y, out_y = info["in_y"], info["out_y"]
        side = info["side"]

        is_attn = comp.startswith("attn_")
        is_mlp = comp.startswith("mlp_")
        is_emb = comp in ("embedding", "pos_embedding")

        if is_attn:
            fill, stroke, tcol = "#fff0f0", "#d06060", "#a03030"
        elif is_mlp:
            fill, stroke, tcol = "#f0f0ff", "#6060d0", "#3030a0"
        elif is_emb:
            fill, stroke, tcol = "#f0f0f0", "#aaa", "#666"
        else:
            fill, stroke, tcol = "#fff", "#ccc", "#666"

        if not is_emb:
            # Read stub: stream at in_y → component bottom
            svg.append(f'<path d="M{sx},{in_y} L{cx},{cy + box_h/2}" '
                       f'fill="none" stroke="{stroke}" stroke-width="1" '
                       f'stroke-opacity="0.35"/>')
            svg.append(f'<circle cx="{sx}" cy="{in_y}" r="2" fill="{stroke}" opacity="0.4"/>')
            # Write stub: component top → stream at out_y
            svg.append(f'<path d="M{cx},{cy - box_h/2} L{sx},{out_y}" '
                       f'fill="none" stroke="{stroke}" stroke-width="1" '
                       f'stroke-opacity="0.35"/>')
            svg.append(f'<circle cx="{sx}" cy="{out_y}" r="2" fill="{stroke}" opacity="0.4"/>')

        rx, ry = cx - box_w/2, cy - box_h/2
        svg.append(f'<rect x="{rx}" y="{ry}" width="{box_w}" height="{box_h}" '
                   f'rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2" '
                   f'filter="url(#sh)"/>')
        svg.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="{tcol}" '
                   f'font-family="Helvetica,sans-serif" font-size="9" '
                   f'font-weight="600">{_esc(_short(comp))}</text>')

    # ── Pre-scan: find cross-position attention hops to place ghost boxes ──
    ghost_boxes = {}  # (component, source_pos) → {cx, cy, stream_x}
    for vp in graph.paths:
        hops = list(reversed(vp.hops))
        for i in range(len(hops) - 1):
            h_lo = hops[i]
            h_hi = hops[i + 1]
            if h_lo.position != h_hi.position and h_hi.is_attn:
                # h_hi is at query pos, need a ghost at source pos
                gkey = (h_hi.component, h_lo.position)
                if gkey not in ghost_boxes and h_lo.position in pos_to_col:
                    col = pos_to_col[h_lo.position]
                    gsx = x_of(col)
                    gcy = y_of(h_hi.layer)  # same layer as the real head
                    gcx = gsx - branch
                    ghost_boxes[gkey] = {
                        "cx": gcx, "cy": gcy, "stream_x": gsx,
                        "in_y": gcy + detour_h, "out_y": gcy - detour_h,
                    }
                    # Extend stream for ghost position
                    glayer = int(h_hi.layer)
                    if glayer > pos_max_layer[h_lo.position]:
                        pos_max_layer[h_lo.position] = glayer

    # Draw ghost boxes (dashed border, lighter)
    for (comp, pos), g in ghost_boxes.items():
        cx, cy, sx = g["cx"], g["cy"], g["stream_x"]
        stroke = "#d06060"
        # Stub from stream
        svg.append(f'<path d="M{sx},{g["in_y"]} L{cx},{cy + box_h/2}" '
                   f'fill="none" stroke="{stroke}" stroke-width="1" stroke-opacity="0.25"/>')
        svg.append(f'<circle cx="{sx}" cy="{g["in_y"]}" r="2" fill="{stroke}" opacity="0.3"/>')
        # Ghost box (dashed)
        rx, ry = cx - box_w/2, cy - box_h/2
        svg.append(f'<rect x="{rx}" y="{ry}" width="{box_w}" height="{box_h}" '
                   f'rx="5" fill="#fff8f8" stroke="{stroke}" stroke-width="1" '
                   f'stroke-dasharray="3,2" stroke-opacity="0.5" filter="url(#sh)"/>')
        svg.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="#c06060" '
                   f'font-family="Helvetica,sans-serif" font-size="9" '
                   f'font-weight="600" opacity="0.6">{_esc(_short(comp))}</text>')

    # ── Paths ──
    for vp in graph.paths:
        color = vp.color or "#c0392b"
        op = min(0.85, max(0.3, abs(vp.score) / 25))
        sw = "2.5"
        safe = _css_safe(color)

        hops = list(reversed(vp.hops))  # bottom-to-top

        for i in range(len(hops) - 1):
            h_lo = hops[i]
            h_hi = hops[i + 1]

            k_lo = (h_lo.component, h_lo.position)
            k_hi = (h_hi.component, h_hi.position)
            if k_lo not in comp_info or k_hi not in comp_info:
                continue

            lo = comp_info[k_lo]
            hi = comp_info[k_hi]

            lo_out_y = lo["out_y"] if lo["side"] != "center" else lo["cy"] - box_h/2
            hi_in_y = hi["in_y"] if hi["side"] != "center" else hi["cy"] + box_h/2
            lo_sx = lo["stream_x"]
            hi_sx = hi["stream_x"]

            # Write-out from lo
            if lo["side"] != "center":
                svg.append(f'<path d="M{lo["cx"]},{lo["cy"] - box_h/2} L{lo_sx},{lo_out_y}" '
                           f'fill="none" stroke="{color}" stroke-width="{sw}" '
                           f'stroke-opacity="{op:.2f}"/>')

            if h_lo.position == h_hi.position:
                # Same position: up the stream
                svg.append(f'<line x1="{lo_sx}" y1="{lo_out_y}" x2="{hi_sx}" y2="{hi_in_y}" '
                           f'stroke="{color}" stroke-width="{sw}" stroke-opacity="{op:.2f}"/>')
            else:
                # Cross-position attention: route through ghost box
                gkey = (h_hi.component, h_lo.position)
                g = ghost_boxes.get(gkey)
                if g:
                    mode = h_hi.mode or ""
                    label_op = min(op + 0.15, 1)
                    # Up source stream to ghost in_y
                    svg.append(f'<line x1="{lo_sx}" y1="{lo_out_y}" '
                               f'x2="{lo_sx}" y2="{g["in_y"]}" '
                               f'stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}"/>')
                    # Into ghost box
                    svg.append(f'<path d="M{lo_sx},{g["in_y"]} L{g["cx"]},{g["cy"] + box_h/2}" '
                               f'fill="none" stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}"/>')
                    # Horizontal: ghost → real head
                    svg.append(f'<line x1="{g["cx"] + box_w/2}" y1="{g["cy"]}" '
                               f'x2="{hi["cx"] - box_w/2}" y2="{hi["cy"]}" '
                               f'stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}" stroke-dasharray="5,3" '
                               f'marker-end="url(#a-{safe})"/>')
                    # Mode label on the horizontal line
                    if mode:
                        mid_x = (g["cx"] + box_w/2 + hi["cx"] - box_w/2) / 2
                        mid_y = (g["cy"] + hi["cy"]) / 2 - 6
                        svg.append(f'<text x="{mid_x}" y="{mid_y}" text-anchor="middle" '
                                   f'fill="{color}" font-family="Helvetica,sans-serif" '
                                   f'font-size="9" font-weight="bold" '
                                   f'opacity="{label_op:.2f}">{mode}</text>')
                else:
                    # Fallback: simple curve if no ghost
                    cross_y = (lo_out_y + hi_in_y) / 2
                    svg.append(f'<line x1="{lo_sx}" y1="{lo_out_y}" '
                               f'x2="{lo_sx}" y2="{cross_y}" '
                               f'stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}"/>')
                    ctrl = abs(hi_sx - lo_sx) * 0.2
                    svg.append(f'<path d="M{lo_sx},{cross_y} C{lo_sx},{cross_y - ctrl} '
                               f'{hi_sx},{cross_y + ctrl} {hi_sx},{cross_y}" '
                               f'fill="none" stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}"/>')
                    svg.append(f'<line x1="{hi_sx}" y1="{cross_y}" '
                               f'x2="{hi_sx}" y2="{hi_in_y}" '
                               f'stroke="{color}" stroke-width="{sw}" '
                               f'stroke-opacity="{op:.2f}"/>')

            # Read-in to hi (skip for cross-position — already connected via ghost)
            if hi["side"] != "center" and h_lo.position == h_hi.position:
                svg.append(f'<path d="M{hi_sx},{hi_in_y} L{hi["cx"]},{hi["cy"] + box_h/2}" '
                           f'fill="none" stroke="{color}" stroke-width="{sw}" '
                           f'stroke-opacity="{op:.2f}" '
                           f'marker-end="url(#a-{safe})"/>')

            # Mode tag for same-position hops only
            if h_lo.position == h_hi.position:
                mode = h_hi.mode or ""
                if mode:
                    tag_x = hi["cx"] - box_w/2 - 16
                    tag_y = hi["cy"] + 4
                    svg.append(f'<text x="{tag_x}" y="{tag_y}" fill="{color}" '
                               f'font-family="Helvetica,sans-serif" font-size="8" '
                               f'font-weight="bold" opacity="{min(op+0.15,1):.2f}">'
                               f'[{mode}]</text>')

    # ── Legend ──
    lx, ly = margin, 22
    for i, vp in enumerate(graph.paths):
        px = lx + i * 220
        if px + 200 > total_width: break
        color = vp.color or "#c0392b"
        label = vp.label or f"path {i+1}"
        score = f" {vp.score:+.1f}%" if vp.score else ""
        svg.append(f'<line x1="{px}" y1="{ly}" x2="{px+16}" y2="{ly}" '
                   f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        svg.append(f'<text x="{px+22}" y="{ly+4}" fill="#555" '
                   f'font-family="Helvetica,sans-serif" font-size="9">'
                   f'{_esc(label)}{score}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def _short(c):
    if c == "embedding": return "embed"
    if c == "pos_embedding": return "pos_emb"
    m = re.match(r"attn_(\d+)_head_(\d+)", c)
    if m: return f"A{m.group(1)}.H{m.group(2)}"
    m = re.match(r"mlp_(\d+)", c)
    if m: return f"MLP{m.group(1)}"
    return c[:10]

def _esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _css_safe(c):
    return c.replace("#","c").replace("(","").replace(")","").replace(",","")
