"""
render_tikz — Publication-quality TikZ circuit-diagram renderer.

Produces standalone .tex compilable with pdflatex, or a bare
tikzpicture block for \\input{} in a larger document.
"""

from __future__ import annotations
import json, re, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    m = _HEX.match(h)
    if not m:
        return (100, 100, 100)
    s = m.group(1)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def _tikz_color_def(name: str, hexc: str) -> str:
    r, g, b = _hex_to_rgb(hexc)
    return f"\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}"

def _short_name(c: str) -> str:
    if c == "embedding":
        return "embed"
    if c == "pos_embedding":
        return r"pos\_emb"
    m = re.match(r"attn_(\d+)_head_(\d+)", c)
    if m:
        return f"A{m.group(1)}.H{m.group(2)}"
    m = re.match(r"mlp_(\d+)", c)
    if m:
        return f"MLP {m.group(1)}"
    return c[:10].replace("_", r"\_")

def _edge_label(c: str) -> str:
    m = re.search(r"head_(\d+)", c)
    if m:
        return f"H{m.group(1)}"
    m = re.match(r"mlp_(\d+)", c)
    if m:
        return f"M{m.group(1)}"
    return _short_name(c)

def _node_name(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", key)

def _esc(s: str) -> str:
    for ch, rep in [("_", r"\_"), ("&", r"\&"), ("#", r"\#"),
                    ("$", r"\$"), ("%", r"\%"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(ch, rep)
    return s


def render_tikz(data: dict, *,
                standalone: bool = True,
                width_cm: float = 18.0,
                layer_height_cm: float = 1.6,
                pos_spacing_cm: float = 3.8,
                box_w: float = 1.8,
                box_h: float = 0.62,
                branch: float = 1.4,
                detour: float = 0.38,
                lane_gap: float = 0.14,
                card_offset: float = 0.55,
                ) -> str:
    """Render circuit paths as TikZ.

    Args:
        data: dict from CircuitGraph.to_dict()
        standalone: if True, emit full compilable document.
                    if False, emit only pc color defs + tikzpicture
                    (requires shared preamble via \\input{path_figure_preamble}).
    """

    arch = data["architecture"]
    num_layers = arch["num_layers"]
    positions = data["active_positions"]
    n_cols = len(positions)
    pos_to_col = {p: i for i, p in enumerate(positions)}

    span = (n_cols - 1) * pos_spacing_cm
    x_origin = (width_cm - span) / 2

    def x_of(col):
        return x_origin + col * pos_spacing_cm

    def y_of(layer):
        return (layer + 1) * layer_height_cm + 1.0

    bw2, bh2 = box_w / 2, box_h / 2

    # ── collect component positions ──
    layer_pos_comps: Dict[str, List[str]] = {}
    seen = set()
    for path in data["paths"]:
        for h in path["hops"]:
            key = h["component"] + "@" + str(h["position"])
            if key in seen or h["position"] not in pos_to_col:
                continue
            seen.add(key)
            is_emb = h["component"] in ("embedding", "pos_embedding")
            li = -1 if is_emb else h["layer"]
            lp = f"{li}:{h['position']}"
            layer_pos_comps.setdefault(lp, []).append(key)

    comp_info: Dict[str, dict] = {}
    for path in data["paths"]:
        for h in path["hops"]:
            key = h["component"] + "@" + str(h["position"])
            if key in comp_info or h["position"] not in pos_to_col:
                continue
            col = pos_to_col[h["position"]]
            sx = x_of(col)
            cy = y_of(h["layer"])
            is_emb = h["component"] in ("embedding", "pos_embedding")
            is_attn = h["component"].startswith("attn_")
            if is_emb:
                siblings = layer_pos_comps.get(f"-1:{h['position']}", [key])
                idx = siblings.index(key) if key in siblings else 0
                cx = sx - card_offset * idx
            else:
                li = h["layer"]
                siblings = layer_pos_comps.get(f"{li}:{h['position']}", [key])
                idx = siblings.index(key) if key in siblings else 0
                cx = sx - branch - card_offset * idx
            comp_info[key] = dict(
                cx=cx, cy=cy, sx=sx,
                in_y=cy - detour, out_y=cy + detour,
                is_emb=is_emb, is_attn=is_attn,
                comp=h["component"], pos=h["position"],
                sib_idx=idx,
            )

    # ── ghost boxes ──
    ghosts: Dict[str, dict] = {}
    for path in data["paths"]:
        hops = list(reversed(path["hops"]))
        for i in range(len(hops) - 1):
            lo, hi = hops[i], hops[i + 1]
            if lo["position"] != hi["position"] and hi["component"].startswith("attn_"):
                gk = hi["component"] + "@" + str(lo["position"]) + "_ghost"
                if gk not in ghosts and lo["position"] in pos_to_col:
                    col = pos_to_col[lo["position"]]
                    sx = x_of(col)
                    cy = y_of(hi["layer"])
                    ghosts[gk] = dict(cx=sx - branch, cy=cy, sx=sx,
                                      in_y=cy - detour, out_y=cy + detour,
                                      comp=hi["component"], pos=lo["position"])

    # ── lane assignment ──
    pos_path_indices: Dict[int, List[int]] = {}
    ghost_target_paths: Dict[str, List[int]] = {}
    for pi, path in enumerate(data["paths"]):
        touched: Set[int] = set()
        for h in path["hops"]:
            touched.add(h["position"])
        hops = list(reversed(path["hops"]))
        for i in range(len(hops) - 1):
            lo, hi = hops[i], hops[i + 1]
            if lo["position"] != hi["position"] and hi["component"].startswith("attn_"):
                touched.add(lo["position"])
                tk = hi["component"] + "@" + str(hi["position"])
                ghost_target_paths.setdefault(tk, [])
                if pi not in ghost_target_paths[tk]:
                    ghost_target_paths[tk].append(pi)
        for pos in touched:
            pos_path_indices.setdefault(pos, [])
            if pi not in pos_path_indices[pos]:
                pos_path_indices[pos].append(pi)

    def stream_off(pos, pi):
        idx_list = pos_path_indices.get(pos, [pi])
        if pi not in idx_list:
            return 0
        return (idx_list.index(pi) - (len(idx_list) - 1) / 2) * lane_gap

    ghost_area_links: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for pi, path in enumerate(data["paths"]):
        hops = list(reversed(path["hops"]))
        for i in range(len(hops) - 1):
            lo, hi = hops[i], hops[i + 1]
            if lo["position"] != hi["position"] and hi["component"].startswith("attn_"):
                area = (hi["position"], round(hi["layer"]))
                hk = hi["component"] + "@" + str(hi["position"])
                ghost_area_links.setdefault(area, []).append((pi, hk))

    _ghost_area_gap = lane_gap

    def ghost_area_off(hi_pos, hi_layer, pi):
        area = (hi_pos, round(hi_layer))
        group = ghost_area_links.get(area, [])
        idx = next((j for j, (p, _) in enumerate(group) if p == pi), 0)
        n = len(group)
        return (idx - (n - 1) / 2) * _ghost_area_gap

    top_comps_by_pos: Dict[int, List[dict]] = {}
    for pi, path in enumerate(data["paths"]):
        if path["hops"]:
            h = path["hops"][0]
            top_comps_by_pos.setdefault(h["position"], []).append(
                dict(comp=h["component"],
                     key=h["component"] + "@" + str(h["position"]), pi=pi))

    pos_max_l = {p: -1 for p in positions}
    for info in comp_info.values():
        layer_approx = int(round((info["cy"] - 1.0) / layer_height_cm - 1))
        if layer_approx > pos_max_l.get(info["pos"], -1):
            pos_max_l[info["pos"]] = layer_approx

    placed_modes: Dict[str, List[str]] = {}
    def get_mode_offset(comp_key, mode):
        placed_modes.setdefault(comp_key, [])
        if mode in placed_modes[comp_key]:
            return None
        off = len(placed_modes[comp_key]) * 0.30
        placed_modes[comp_key].append(mode)
        return off

    # ══════════════════════════════════════════════════════════
    O: List[str] = []
    W = O.append

    if standalone:
        W(r"\documentclass[border=6mm]{standalone}")
        W(r"\usepackage{tikz}")
        W(r"\usetikzlibrary{arrows.meta, calc, positioning, backgrounds,"
          r" fit, shapes.geometric, fadings, decorations.markings}")
        W(r"\usepackage{xcolor}")
        W("")
        W("% Component palette")
        W(_tikz_color_def("attnFill", "#dae7f6"))
        W(_tikz_color_def("attnHi", "#eaf1fb"))
        W(_tikz_color_def("attnStroke", "#5a85c0"))
        W(_tikz_color_def("attnText", "#2e4a78"))
        W(_tikz_color_def("mlpFill", "#faf0d8"))
        W(_tikz_color_def("mlpHi", "#fef8ec"))
        W(_tikz_color_def("mlpStroke", "#c8a030"))
        W(_tikz_color_def("mlpText", "#7a6420"))
        W(_tikz_color_def("embFill", "#eaeaea"))
        W(_tikz_color_def("embHi", "#f5f5f5"))
        W(_tikz_color_def("embStroke", "#aaaaaa"))
        W(_tikz_color_def("embText", "#606060"))
        W(_tikz_color_def("ghostFill", "#e6ecf5"))
        W(_tikz_color_def("ghostStroke", "#8aa4cc"))
        W(_tikz_color_def("ghostText", "#8898b8"))
        W(_tikz_color_def("tgtFill", "#eee8f8"))
        W(_tikz_color_def("tgtHi", "#f8f4ff"))
        W(_tikz_color_def("tgtStroke", "#8878a8"))
        W(_tikz_color_def("tgtText", "#5848a0"))
        W(_tikz_color_def("shCol", "#c8c8cc"))
        W(_tikz_color_def("ruleG", "#e8e8e8"))
        W(_tikz_color_def("labG", "#bbbbbb"))
        W(_tikz_color_def("layG", "#cccccc"))
        W(_tikz_color_def("tokC", "#484848"))
        W(_tikz_color_def("posC", "#b0b0b0"))
        W("")
        W(r"\tikzset{")
        W(f"  cb/.style={{minimum width={box_w}cm, minimum height={box_h}cm,")
        W(r"    rounded corners=3pt, inner sep=0pt, line width=0.5pt,")
        W(r"    font=\sffamily\scriptsize\bfseries, align=center},")
        W(f"  cs/.style={{minimum width={box_w}cm, minimum height={box_h}cm,")
        W(r"    rounded corners=3pt, fill=shCol, opacity=0.22},")
        W(r"  atn/.style={cb, top color=attnHi, bottom color=attnFill, draw=attnStroke, text=attnText},")
        W(r"  mlp/.style={cb, top color=mlpHi, bottom color=mlpFill, draw=mlpStroke, text=mlpText},")
        W(r"  emb/.style={cb, top color=embHi, bottom color=embFill, draw=embStroke, text=embText},")
        W(r"  gho/.style={cb, fill=ghostFill, draw=ghostStroke, text=ghostText, dash pattern=on 2.5pt off 1.8pt, draw opacity=0.55, text opacity=0.7},")
        W(r"  tgt/.style={cb, top color=tgtHi, bottom color=tgtFill, draw=tgtStroke, text=tgtText, line width=0.7pt},")
        W(r"  ll/.style={font=\sffamily\scriptsize, text=layG, anchor=east},")
        W(r"  tl/.style={font=\sffamily\small\bfseries, text=tokC, anchor=north},")
        W(r"  pl/.style={font=\sffamily\tiny, text=posC, anchor=north},")
        W(r"  mt/.style={font=\sffamily\tiny\bfseries, opacity=0.8},")
        W(r"  ar/.style={-{Stealth[length=3.5pt, width=2.8pt]}},")
        W(r"  dar/.style={ar, dash pattern=on 4pt off 2.5pt},")
        W(r"}")
        W("")
        W(r"\begin{document}")

    # Per-figure path colors (always needed)
    W("% Path colours")
    pcn = []
    used: List[str] = []
    for pi, path in enumerate(data["paths"]):
        c = path.get("color") or "#c0392b"
        nm = f"pc{pi}"
        pcn.append(nm)
        if c not in used:
            W(_tikz_color_def(nm, c))
            used.append(c)
        else:
            W(f"\\colorlet{{{nm}}}{{pc{used.index(c)}}}")

    W(r"\begin{tikzpicture}[x=1cm, y=1cm]")
    W("")

    x_min = x_of(0) - branch - 1.5
    x_max = x_of(n_cols - 1) + 1.5 if n_cols > 0 else width_cm
    y_max = y_of(max(pos_max_l.values()) if pos_max_l else 0) + 2.5

    active_layers: Set[int] = set()
    for path in data["paths"]:
        for h in path["hops"]:
            if h["component"] not in ("embedding", "pos_embedding"):
                active_layers.add(int(h["layer"]))

    lx = x_of(0) - branch - 0.9
    W("  % Layers")
    for layer in sorted(active_layers):
        ly = y_of(layer)
        W(f"  \\node[ll] at ({lx:.2f},{ly:.2f}) {{L{layer}}};")
        W(f"  \\draw[ruleG, line width=0.25pt] ({x_min:.2f},{ly:.2f}) -- ({x_max:.2f},{ly:.2f});")

    has_embed = any(h["component"] == "embedding"
                    for p in data["paths"] for h in p["hops"])
    if has_embed:
        ey = y_of(-1)
        W(f"  \\node[ll] at ({lx:.2f},{ey:.2f}) {{emb}};")
        W(f"  \\draw[ruleG, line width=0.25pt] ({x_min:.2f},{ey:.2f}) -- ({x_max:.2f},{ey:.2f});")
    W("")

    # ── tokens ──
    raw_tokens = data.get("tokens")
    tok_map: Dict[int, str] = {}
    if isinstance(raw_tokens, list):
        for idx, t in enumerate(raw_tokens):
            if t is not None:
                tok_map[idx] = str(t)
    elif isinstance(raw_tokens, dict):
        for k, v in raw_tokens.items():
            tok_map[int(k)] = str(v) if v is not None else ""

    W("  % Tokens")
    for i, pos in enumerate(positions):
        cx = x_of(i)
        ey = y_of(-1)
        tok = tok_map.get(pos, "")
        if tok:
            tok = tok.replace("\u0120", " ").strip()[:14]
        W(f"  \\node[tl] at ({cx:.2f},{ey - bh2 - 0.22:.2f}) {{{_esc(tok)}}};")
        W(f"  \\node[pl] at ({cx:.2f},{ey - bh2 - 0.50:.2f}) {{pos {pos}}};")
        if i < len(positions) - 1 and positions[i + 1] - pos > 1:
            mx = (x_of(i) + x_of(i + 1)) / 2
            W(f"  \\node[text=labG] at ({mx:.2f},{ey - bh2 - 0.22:.2f}) {{$\\cdots$}};")
    W("")

    rightmost = positions[-1] if positions else 0
    target_y = None
    if positions and not data.get("root"):
        tp = rightmost
        tx = x_of(pos_to_col[tp])
        top_l = max(pos_max_l.get(tp, 0), 0)
        target_y = y_of(top_l) + detour + 1.1
        lbl = _esc((data.get("target_token") or "target").strip())
        W("  % Target")
        W(f"  \\node[cs] at ({tx + 0.03:.2f},{target_y - 0.05:.2f}) {{}};")
        W(f"  \\node[tgt] (tgt) at ({tx:.2f},{target_y:.2f}) {{{lbl}}};")
        top_at = top_comps_by_pos.get(tp, [])
        n_top = len(top_at)
        for ti, entry in enumerate(top_at):
            info = comp_info.get(entry["key"])
            if not info:
                continue
            cn = pcn[entry["pi"]]
            off = (ti - (n_top - 1) / 2) * lane_gap
            sx = info["sx"] + off
            oy = info["out_y"]
            W(f"  \\draw[{cn}, line width=0.9pt, opacity=0.30]"
              f" ({info['cx'] + off:.3f},{info['cy'] + bh2:.3f})"
              f" .. controls ({sx:.3f},{info['cy'] + bh2 + 0.15:.3f})"
              f" and ({sx:.3f},{oy:.3f}) .. ({sx:.3f},{oy:.3f})"
              f" -- ({tx + off:.3f},{target_y - bh2:.3f});")
        W("")

    W("  % Shadows")
    sorted_comps = sorted(comp_info.items(),
                          key=lambda kv: -(kv[1].get("sib_idx", 0)))
    for key, info in sorted_comps:
        W(f"  \\node[cs] at ({info['cx'] + 0.03:.3f},{info['cy'] - 0.05:.3f}) {{}};")
    W("")

    W("  % Components")
    for key, info in sorted_comps:
        cx, cy = info["cx"], info["cy"]
        is_root = data.get("root") and info["comp"] == data["root"]
        style = ("atn" if is_root or info["is_attn"] else
                 "emb" if info["is_emb"] else "mlp")
        nn = _node_name(key)
        sib = info.get("sib_idx", 0)
        if sib == 0:
            W(f"  \\node[{style}] ({nn}) at ({cx:.3f},{cy:.3f}) {{{_short_name(info['comp'])}}};")
        else:
            W(f"  \\node[{style}] ({nn}) at ({cx:.3f},{cy:.3f}) {{}};")
            elbl = _edge_label(info["comp"])
            tcol = "attnText" if info["is_attn"] else ("embText" if info["is_emb"] else "mlpText")
            W(f"  \\node[font=\\sffamily\\tiny\\bfseries, text={tcol}, anchor=west]"
              f" at ({cx - bw2 + 0.06:.3f},{cy:.3f}) {{{elbl}}};")
    W("")

    if ghosts:
        W("  % Ghosts")
        for gk, g in ghosts.items():
            W(f"  \\node[gho] ({_node_name(gk)}) at ({g['cx']:.3f},{g['cy']:.3f})"
              f" {{{_short_name(g['comp'])}}};")
        W("")

    W("  % Paths")
    for pi, path in enumerate(data["paths"]):
        cn = pcn[pi]
        op = min(0.80, max(0.35, abs(path.get("score", 0)) / 18))
        hops = list(reversed(path["hops"]))

        W(f"  \\begin{{scope}}[{cn}, line width=1.3pt, opacity={op:.2f}]")

        for i in range(len(hops) - 1):
            lo, hi = hops[i], hops[i + 1]
            k_lo = lo["component"] + "@" + str(lo["position"])
            k_hi = hi["component"] + "@" + str(hi["position"])
            i_lo = comp_info.get(k_lo)
            i_hi = comp_info.get(k_hi)
            if not i_lo or not i_hi:
                continue

            lo_off = stream_off(lo["position"], pi)
            hi_off = stream_off(hi["position"], pi)
            lo_out = (i_lo["cy"] + bh2) if i_lo["is_emb"] else i_lo["out_y"]
            hi_in = (i_hi["cy"] - bh2) if i_hi["is_emb"] else i_hi["in_y"]

            if lo["position"] == hi["position"]:
                if not i_lo["is_emb"]:
                    ax = i_lo["cx"] + lo_off
                    bx = i_lo["sx"] + lo_off
                    lt = i_lo["cy"] + bh2
                    W(f"    \\draw ({ax:.3f},{lt:.3f})"
                      f" .. controls ({ax:.3f},{lt + 0.15:.3f})"
                      f" and ({bx:.3f},{lo_out + 0.08:.3f})"
                      f" .. ({bx:.3f},{lo_out:.3f});")
                sx_lo = i_lo["sx"] + lo_off
                sx_hi = i_hi["sx"] + hi_off
                W(f"    \\draw ({sx_lo:.3f},{lo_out:.3f}) -- ({sx_hi:.3f},{hi_in:.3f});")
                if not i_hi["is_emb"]:
                    bx = i_hi["sx"] + hi_off
                    ax = i_hi["cx"] + hi_off
                    hb = i_hi["cy"] - bh2
                    W(f"    \\draw[ar] ({bx:.3f},{hi_in:.3f})"
                      f" .. controls ({bx:.3f},{hb - 0.08:.3f})"
                      f" and ({ax:.3f},{hb - 0.15:.3f})"
                      f" .. ({ax:.3f},{hb:.3f});")
                if hi.get("mode"):
                    moff = get_mode_offset(k_hi, hi["mode"])
                    if moff is not None:
                        mx = i_hi["cx"] - bw2 - 0.10
                        my = i_hi["cy"] - bh2 + 0.08 + moff
                        W(f"    \\node[mt, anchor=east, text={cn}]"
                          f" at ({mx:.3f},{my:.3f}) {{{hi['mode']}}};")
            else:
                gk = hi["component"] + "@" + str(lo["position"]) + "_ghost"
                ghost = ghosts.get(gk)
                if ghost:
                    gl_off = ghost_area_off(hi["position"], hi["layer"], pi)
                    has_gap = abs(i_lo["cy"] - ghost["cy"]) > 0.05
                    ghost_in = ghost["cy"] - detour

                    if has_gap:
                        if not i_lo["is_emb"]:
                            ax = i_lo["cx"] + lo_off
                            bx = i_lo["sx"] + lo_off
                            lt = i_lo["cy"] + bh2
                            W(f"    \\draw ({ax:.3f},{lt:.3f})"
                              f" .. controls ({ax:.3f},{lt + 0.15:.3f})"
                              f" and ({bx:.3f},{lo_out + 0.08:.3f})"
                              f" .. ({bx:.3f},{lo_out:.3f});")
                        sx_lo = i_lo["sx"] + lo_off
                        gsx = ghost["sx"] + lo_off
                        W(f"    \\draw ({sx_lo:.3f},{lo_out:.3f})"
                          f" -- ({gsx:.3f},{ghost_in:.3f});")

                    gsx = ghost["sx"] + lo_off
                    gcx = ghost["cx"] + lo_off
                    ghb = ghost["cy"] - bh2
                    W(f"    \\draw[ar] ({gsx:.3f},{ghost_in:.3f})"
                      f" .. controls ({gsx:.3f},{ghb - 0.08:.3f})"
                      f" and ({gcx:.3f},{ghb - 0.15:.3f})"
                      f" .. ({gcx:.3f},{ghb:.3f});")

                    gy = ghost["cy"] + gl_off
                    W(f"    \\draw[dar]"
                      f" ({ghost['cx'] + bw2:.3f},{gy:.3f})"
                      f" -- ({i_hi['cx'] - bw2:.3f},{i_hi['cy'] + gl_off:.3f});")

                    if hi.get("mode"):
                        ck = hi["component"] + "@" + str(hi["position"]) + "_from_" + str(lo["position"])
                        moff = get_mode_offset(ck, hi["mode"])
                        if moff is not None:
                            mx = (ghost["cx"] + bw2 + i_hi["cx"] - bw2) / 2
                            my = gy + bh2 + 0.12 + moff
                            W(f"    \\node[mt, text={cn}]"
                              f" at ({mx:.3f},{my:.3f}) {{{hi['mode']}}};")

        W(r"  \end{scope}")
    W("")

    W("  % Legend")
    legend_y = y_max + 0.45
    np_ = len(data["paths"])
    lw = np_ * 4.2 + 0.5
    W(f"  \\fill[white, rounded corners=4pt, draw=ruleG, line width=0.3pt]"
      f" ({x_min + 0.2:.2f},{legend_y - 0.28:.2f})"
      f" rectangle ({x_min + lw:.2f},{legend_y + 0.30:.2f});")
    for pi, path in enumerate(data["paths"]):
        cn = pcn[pi]
        lx_ = x_min + 0.6 + pi * 4.2
        label = _esc(path.get("label") or f"path {pi + 1}")
        ss = ""
        if path.get("score"):
            s = path["score"]
            ss = f" {'+'if s > 0 else ''}{s:.1f}\\%"
        W(f"  \\draw[{cn}, line width=2pt, line cap=round] ({lx_:.2f},{legend_y:.2f})"
          f" -- ++(0.35,0);")
        W(f"  \\node[font=\\sffamily\\tiny, text=tokC, anchor=west]"
          f" at ({lx_ + 0.50:.2f},{legend_y:.2f}) {{{label}{ss}}};")
    W("")

    if data.get("target_token"):
        t = r"Circuit $\rightarrow$ \texttt{" + _esc(data["target_token"].strip()) + "}"
        W(f"  \\node[font=\\sffamily\\small\\bfseries, text=tokC, anchor=west]"
          f" at ({x_min + 0.2:.2f},{legend_y + 0.72:.2f}) {{{t}}};")
    W("")

    W(r"\end{tikzpicture}")

    if standalone:
        W(r"\end{document}")

    return "\n".join(O)