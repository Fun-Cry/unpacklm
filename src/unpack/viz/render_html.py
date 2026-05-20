"""
unpack.viz.render_html - Interactive circuit visualization.

Pure vanilla JS + inline SVG. No external dependencies.
Curved stubs, muted palette, smooth connections.
"""

from __future__ import annotations
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from unpack.viz.graph import CircuitGraph


def render_html(graph: "CircuitGraph", width=900, height=None) -> str:
    data = graph.to_dict()
    data_json = json.dumps(data)
    
    if height is None:
        n_layers = data["architecture"]["num_layers"]
        height = (n_layers + 3) * 52 + 160

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #ffffff; font-family: 'Inter', 'SF Pro Text', -apple-system, sans-serif; }}
.circuit-svg {{ display: block; margin: 20px auto; }}
.stream {{ stroke: #e8e8e8; stroke-width: 2.5; stroke-linecap: round; }}
.stream-arrow {{ fill: #ddd; }}
.comp-box {{ stroke-width: 1; }}
.stub {{ fill: none; stroke-width: 0.8; stroke-opacity: 0.3; }}
.stub-dot {{ opacity: 0.3; }}
.path-seg {{ fill: none; stroke-linecap: round; stroke-linejoin: round; }}
.path-seg.ghost-link {{ stroke-dasharray: 6,4; }}
.legend-item {{ cursor: pointer; user-select: none; }}
.legend-item:hover .legend-text {{ opacity: 1; }}
.path-group.dimmed .path-seg,
.path-group.dimmed .mode-label {{ opacity: 0.04 !important; }}
.legend-item.dimmed .legend-line {{ opacity: 0.15; }}
.legend-item.dimmed .legend-text {{ opacity: 0.3; }}
</style>
</head><body>
<script>
const DATA = {data_json};
const W = {width}, H = {height};
const NS = "http://www.w3.org/2000/svg";
const MARGIN = {{ top: 55, right: 60, bottom: 60, left: 80 }};
const BOX_W = 62, BOX_H = 22, BRANCH = BOX_W/2 + 20, DETOUR = 14;
const LANE_GAP = 7; // px offset per parallel path
const CURVE_R = 8; // radius for curved stubs

const arch = DATA.architecture;
const numLayers = arch.num_layers;
const positions = DATA.active_positions;
const nCols = positions.length;
const posToCol = {{}};
positions.forEach((p,i) => posToCol[p] = i);

const layerH = (H - MARGIN.top - MARGIN.bottom - 50) / (numLayers + 2);
const posSpacing = Math.min(170, (W - MARGIN.left - MARGIN.right) / Math.max(nCols - 1, 1));

function yOf(layer) {{
  const row = numLayers - (layer + 1);
  return MARGIN.top + 45 + row * layerH;
}}
function xOf(col) {{
  const span = (nCols - 1) * posSpacing;
  return (W - span) / 2 + col * posSpacing;
}}

function mkSvg(tag, attrs) {{
  const el = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs || {{}})) el.setAttribute(k, String(v));
  return el;
}}
function mkText(x, y, text, attrs) {{
  const el = mkSvg("text", {{ x, y, ...attrs }});
  el.textContent = text;
  return el;
}}

const svg = mkSvg("svg", {{ width: W, height: H, class: "circuit-svg" }});
document.body.appendChild(svg);

// Defs
const defs = mkSvg("defs");
svg.appendChild(defs);

// Subtle shadow
const filter = mkSvg("filter", {{ id: "sh", x: "-6%", y: "-6%", width: "112%", height: "112%" }});
filter.appendChild(mkSvg("feDropShadow", {{ dx: 0, dy: 0.5, stdDeviation: 1.2, "flood-opacity": 0.05 }}));
defs.appendChild(filter);

// Arrow markers
const usedColors = new Set(DATA.paths.map(p => p.color || "#c0392b"));
usedColors.forEach(c => {{
  const id = "arr-" + c.replace("#","c");
  const m = mkSvg("marker", {{ id, viewBox: "0 0 10 10", refX: 5, refY: 5,
    markerWidth: 4, markerHeight: 4, orient: "auto" }});
  m.appendChild(mkSvg("path", {{ d: "M0,2 L10,5 L0,8Z", fill: c, opacity: 0.7 }}));
  defs.appendChild(m);
}});

// Background
svg.appendChild(mkSvg("rect", {{ width: W, height: H, fill: "#ffffff", rx: 0 }}));

// ── Compute component positions ──
const compInfo = {{}};
const posMaxL = {{}}, posMinL = {{}};
positions.forEach(p => {{ posMaxL[p] = -1; posMinL[p] = numLayers; }});

// First pass: collect siblings per (layer, position)
const layerPosComps = {{}};
const seenFirst = new Set();
DATA.paths.forEach(path => path.hops.forEach(h => {{
  const key = h.component + "@" + h.position;
  if (seenFirst.has(key) || posToCol[h.position] === undefined) return;
  seenFirst.add(key);
  const isEmb = h.component === "embedding" || h.component === "pos_embedding";
  const li = isEmb ? -1 : h.layer;
  const lpKey = li + ":" + h.position;
  if (!layerPosComps[lpKey]) layerPosComps[lpKey] = [];
  layerPosComps[lpKey].push(key);
}}));

// Second pass: assign positions with deck-of-cards stacking
// Small left offset per sibling — arrows connect to each card's cx
const CARD_OFFSET = 22;
DATA.paths.forEach(path => path.hops.forEach(h => {{
  const key = h.component + "@" + h.position;
  if (compInfo[key] || posToCol[h.position] === undefined) return;
  const col = posToCol[h.position];
  const sx = xOf(col), cy = yOf(h.layer);
  const isEmb = h.component === "embedding" || h.component === "pos_embedding";
  let cx;
  if (isEmb) {{
    const lpKey = "-1:" + h.position;
    const siblings = layerPosComps[lpKey] || [key];
    const idx = siblings.indexOf(key);
    cx = sx - CARD_OFFSET * idx;
  }} else {{
    const li = h.layer;
    const lpKey = li + ":" + h.position;
    const siblings = layerPosComps[lpKey] || [key];
    const idx = siblings.indexOf(key);
    cx = sx - BRANCH - CARD_OFFSET * idx;
  }}
  compInfo[key] = {{ cx, cy, sx, inY: cy + DETOUR, outY: cy - DETOUR, isEmb,
    isAttn: h.component.startsWith("attn_"), comp: h.component, pos: h.position,
    sibIdx: isEmb ? (layerPosComps["-1:" + h.position] || [key]).indexOf(key) : (layerPosComps[h.layer + ":" + h.position] || [key]).indexOf(key) }};
  const li = Math.floor(h.layer);
  if (li > posMaxL[h.position]) posMaxL[h.position] = li;
  if (li < posMinL[h.position]) posMinL[h.position] = li;
}}));

// Ghost boxes
const ghosts = {{}};
DATA.paths.forEach(path => {{
  const hops = [...path.hops].reverse();
  for (let i = 0; i < hops.length - 1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    if (lo.position !== hi.position && hi.component.startsWith("attn_")) {{
      const gk = hi.component + "@" + lo.position + "_ghost";
      if (!ghosts[gk] && posToCol[lo.position] !== undefined) {{
        const col = posToCol[lo.position];
        const sx = xOf(col), cy = yOf(hi.layer);
        ghosts[gk] = {{ cx: sx - BRANCH, cy, sx, inY: cy + DETOUR, outY: cy - DETOUR,
          comp: hi.component, pos: lo.position }};
        const li = Math.floor(hi.layer);
        if (li > posMaxL[lo.position]) posMaxL[lo.position] = li;
      }}
    }}
  }}
}});

// Lane assignment
const posPathIndices = {{}};
DATA.paths.forEach((path, pi) => {{
  const touched = new Set();
  path.hops.forEach(h => touched.add(h.position));
  const hops = [...path.hops].reverse();
  for (let i = 0; i < hops.length - 1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    if (lo.position !== hi.position && hi.component.startsWith("attn_")) touched.add(lo.position);
  }}
  touched.forEach(pos => {{
    if (!posPathIndices[pos]) posPathIndices[pos] = [];
    if (!posPathIndices[pos].includes(pi)) posPathIndices[pos].push(pi);
  }});
}});
const ghostTargetPaths = {{}};
DATA.paths.forEach((path, pi) => {{
  const hops = [...path.hops].reverse();
  for (let i = 0; i < hops.length - 1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    if (lo.position !== hi.position && hi.component.startsWith("attn_")) {{
      const tk = hi.component + "@" + hi.position;
      if (!ghostTargetPaths[tk]) ghostTargetPaths[tk] = [];
      if (!ghostTargetPaths[tk].includes(pi)) ghostTargetPaths[tk].push(pi);
    }}
  }}
}});
function streamOff(pos, pi) {{
  const idx = posPathIndices[pos] || [pi];
  return (idx.indexOf(pi) - (idx.length - 1) / 2) * LANE_GAP;
}}
function ghostLinkOff(tk, pi) {{
  const idx = ghostTargetPaths[tk] || [pi];
  return (idx.indexOf(pi) - (idx.length - 1) / 2) * LANE_GAP;
}}

function shortName(c) {{
  if (c === "embedding") return "embed";
  if (c === "pos_embedding") return "pos_emb";
  let m = c.match(/attn_(\\d+)_head_(\\d+)/);
  if (m) return "A"+m[1]+".H"+m[2];
  m = c.match(/mlp_(\\d+)/);
  if (m) return "MLP "+m[1];
  return c.slice(0,10);
}}

// ── Track which positions have path activity (for stream skipping) ──
const posHasPathActivity = {{}};
positions.forEach(p => posHasPathActivity[p] = false);
DATA.paths.forEach(path => {{
  const hops = [...path.hops].reverse();
  for (let i = 0; i < hops.length - 1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    if (lo.position === hi.position) {{
      posHasPathActivity[lo.position] = true;
    }}
  }}
}});
Object.values(ghosts).forEach(g => posHasPathActivity[g.pos] = true);

// Streams removed — colored paths show the flow directly
const rightmostPos = positions[positions.length - 1];

// ── Layer labels ──
const activeL = new Set();
DATA.paths.forEach(p => p.hops.forEach(h => {{
  if (h.component !== "embedding" && h.component !== "pos_embedding") activeL.add(Math.floor(h.layer));
}}));
activeL.forEach(L => svg.appendChild(mkText(MARGIN.left-22, yOf(L)+4, "L"+L,
  {{ "text-anchor":"end", fill:"#ccc", "font-size":"9px", "font-weight":"500" }})));
if (DATA.paths.some(p => p.hops.some(h => h.component==="embedding")))
  svg.appendChild(mkText(MARGIN.left-22, yOf(-1)+4, "emb",
    {{ "text-anchor":"end", fill:"#ccc", "font-size":"9px", "font-weight":"500" }}));

// ── Token labels ──
positions.forEach((pos, i) => {{
  const cx = xOf(i);
  let tok = "";
  if (DATA.tokens && DATA.tokens[pos]) tok = DATA.tokens[pos].replace(/\\u0120/g," ").trim().slice(0,14);
  const embY = yOf(-1);
  svg.appendChild(mkText(cx, embY + BOX_H/2 + 16, tok, {{ "text-anchor":"middle", fill:"#555", "font-size":"11px", "font-weight":"600" }}));
  svg.appendChild(mkText(cx, embY + BOX_H/2 + 28, "pos "+pos, {{ "text-anchor":"middle", fill:"#bbb", "font-size":"8px" }}));
  if (i < positions.length-1 && positions[i+1]-pos > 1) {{
    const mx = (xOf(i)+xOf(i+1))/2;
    svg.appendChild(mkText(mx, embY + BOX_H/2 + 16, "\\u22ef", {{ "text-anchor":"middle", fill:"#ddd", "font-size":"11px" }}));
  }}
}});

// ── Target / root node ──
// Collect top components (first hop of each path in original order)
const topCompsByPos = {{}}; // pos -> list of top component entries
DATA.paths.forEach((path, pi) => {{
  if (path.hops.length > 0) {{
    const h = path.hops[0];
    if (!topCompsByPos[h.position]) topCompsByPos[h.position] = [];
    topCompsByPos[h.position].push({{ comp: h.component, key: h.component+"@"+h.position, pi }});
  }}
}});

let targetNodeY = null;
let rootInfo = null; // compInfo entry for root, if rerooted
if (positions.length) {{
  const tp = rightmostPos;
  const tx = xOf(posToCol[tp]);
  const topL = posMaxL[tp] >= 0 ? posMaxL[tp] : 0;

  if (DATA.root) {{
    // Find root in compInfo
    rootInfo = compInfo[DATA.root + "@" + tp];

    // In rerooted mode: each path's top component feeds into root
    // Add root to compInfo if not there, and mark connections
    if (!rootInfo) {{
      // Root might not be in any path's hops — add it
      const rootLayer = parseInt((DATA.root.match(/_(\\d+)_head/) || ["","0"])[1]);
      const rcy = yOf(rootLayer);
      rootInfo = {{ cx: tx - BRANCH, cy: rcy, sx: tx, inY: rcy + DETOUR, outY: rcy - DETOUR,
        isEmb: false, isAttn: true, comp: DATA.root, pos: tp }};
      compInfo[DATA.root + "@" + tp] = rootInfo;
      if (rootLayer > posMaxL[tp]) posMaxL[tp] = rootLayer;
    }}
  }} else {{
    // Normal mode: target prediction node
    targetNodeY = yOf(topL) - DETOUR - 38;
    const label = DATA.target_token ? DATA.target_token.trim() : "target";

    svg.appendChild(mkSvg("rect", {{ x:tx-BOX_W/2, y:targetNodeY-BOX_H/2, width:BOX_W, height:BOX_H,
      rx:6, fill:"#f8f5ff", stroke:"#9080b0", "stroke-width":"1", filter:"url(#sh)" }}));
    svg.appendChild(mkText(tx, targetNodeY+4, label,
      {{ "text-anchor":"middle", fill:"#6050a0", "font-size":"9px", "font-weight":"600" }}));

    // Parallel lines from top components to target
    const topAtTp = topCompsByPos[tp] || [];
    const nTop = topAtTp.length;
    topAtTp.forEach((entry, idx) => {{
      const info = compInfo[entry.key];
      if (!info) return;
      const color = DATA.paths[entry.pi]?.color || "#999";
      const off = (idx - (nTop - 1) / 2) * LANE_GAP;
      svg.appendChild(mkSvg("path", {{
        d: `M${{info.cx + off}},${{info.cy - BOX_H/2}} Q${{tx + off}},${{info.cy - BOX_H/2}} ${{tx + off}},${{info.outY}}`,
        fill:"none", stroke:color, "stroke-width":"1.5", "stroke-opacity":"0.4"
      }}));
      svg.appendChild(mkSvg("line", {{ x1:tx+off, y1:info.outY, x2:tx+off, y2:targetNodeY+BOX_H/2,
        stroke:color, "stroke-width":"1.5", "stroke-opacity":"0.35" }}));
    }});
  }}
}}

// ── Stub scan ──
const hasLocalInput = new Set();
const hasLocalOutput = new Set();
DATA.paths.forEach(path => {{
  const hops = [...path.hops].reverse();
  for (let i = 0; i < hops.length - 1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    if (lo.position === hi.position) {{
      hasLocalInput.add(hi.component + "@" + hi.position);
      hasLocalOutput.add(lo.component + "@" + lo.position);
    }}
  }}
}});

// Rerooted: top components at root position feed into root
if (DATA.root && rootInfo) {{
  const rootKey = DATA.root + "@" + rootInfo.pos;
  hasLocalInput.add(rootKey);
  // Don't show write-out from root (it's the destination)
  hasLocalOutput.delete(rootKey);
  // Mark top components as having local output (they feed up to root)
  const topAtRoot = topCompsByPos[rootInfo.pos] || [];
  topAtRoot.forEach(entry => {{
    if (entry.comp !== DATA.root) hasLocalOutput.add(entry.key);
  }});
  // Mark root position as active for stream
  posHasPathActivity[rootInfo.pos] = true;
}}

// ── Component boxes ──
const COLORS = {{
  attn:  {{ fill: "#e8f0fa", stroke: "#5080c0", text: "#304878", stubStroke: "#80a0d0" }},
  mlp:   {{ fill: "#fdf5e6", stroke: "#c8a030", text: "#806820", stubStroke: "#d8c060" }},
  emb:   {{ fill: "#f4f4f4", stroke: "#b0b0b0", text: "#666", stubStroke: "#bbb" }},
  ghost: {{ fill: "#edf2fa", stroke: "#80a0d0", text: "#8098b8", stubStroke: "#a0b8d8" }},
  root:  {{ fill: "#dce8f8", stroke: "#3060a0", text: "#1a3868", stubStroke: "#5080c0" }},
}};

function drawComp(cx, cy, comp, type, isRoot, sibIdx) {{
  const c = isRoot ? COLORS.root : COLORS[type];
  const sw = isRoot ? 2 : 1;
  svg.appendChild(mkSvg("rect", {{ x:cx-BOX_W/2, y:cy-BOX_H/2, width:BOX_W, height:BOX_H,
    rx:6, fill:c.fill, stroke:c.stroke, "stroke-width":sw, filter:"url(#sh)" }}));
  if (sibIdx === 0) {{
    // Foreground card: full label centered
    svg.appendChild(mkText(cx, cy+4, shortName(comp),
      {{ "text-anchor":"middle", fill:c.text, "font-size":"9px", "font-weight":"600" }}));
  }} else {{
    // Background card: short label on visible left edge
    let edgeLabel = shortName(comp);
    const hm = comp.match(/head_(\\d+)/);
    if (hm) edgeLabel = "H" + hm[1];
    const mm = comp.match(/mlp_(\\d+)/);
    if (mm) edgeLabel = "M" + mm[1];
    svg.appendChild(mkText(cx - BOX_W/2 + 4, cy+4, edgeLabel,
      {{ "text-anchor":"start", fill:c.text, "font-size":"8px", "font-weight":"600" }}));
  }}
}}

function drawCurvedStub(sx, sy, cx, cy, stroke, direction) {{
  // direction: "in" (stream→component) or "out" (component→stream)
  if (direction === "in") {{
    // From stream point down to component bottom: curve left then down
    svg.appendChild(mkSvg("path", {{
      d: `M${{sx}},${{sy}} Q${{sx}},${{cy + BOX_H/2}} ${{cx}},${{cy + BOX_H/2}}`,
      class: "stub", stroke
    }}));
  }} else {{
    // From component top to stream point above: curve right then up
    svg.appendChild(mkSvg("path", {{
      d: `M${{cx}},${{cy - BOX_H/2}} Q${{sx}},${{cy - BOX_H/2}} ${{sx}},${{sy}}`,
      class: "stub", stroke
    }}));
  }}
  svg.appendChild(mkSvg("circle", {{ cx:sx, cy:sy, r:1.8, fill:stroke, class:"stub-dot" }}));
}}

// Draw components: background cards (high sibIdx) first, foreground (sibIdx=0) last
Object.entries(compInfo)
  .sort((a, b) => (b[1].sibIdx || 0) - (a[1].sibIdx || 0))
  .forEach(([key, info]) => {{
  const type = info.isEmb ? "emb" : info.isAttn ? "attn" : "mlp";
  const isRoot = DATA.root && info.comp === DATA.root;
  drawComp(info.cx, info.cy, info.comp, type, isRoot, info.sibIdx || 0);
}});

// Ghost boxes (box only, no stubs)
Object.entries(ghosts).forEach(([gk, g]) => {{
  svg.appendChild(mkSvg("rect", {{ x:g.cx-BOX_W/2, y:g.cy-BOX_H/2, width:BOX_W, height:BOX_H,
    rx:6, fill:COLORS.ghost.fill, stroke:COLORS.ghost.stroke, "stroke-width":0.8, "stroke-dasharray":"4,3",
    "stroke-opacity":"0.5", filter:"url(#sh)" }}));
  svg.appendChild(mkText(g.cx, g.cy+4, shortName(g.comp),
    {{ "text-anchor":"middle", fill:COLORS.ghost.text, "font-size":"9px", "font-weight":"500", opacity:"0.7" }}));
}});

// ── Paths with curved connections ──
// Track placed mode labels to avoid overlap
const placedModes = {{}}; // "comp@pos" -> [{{mode, x, y}}]
function getModeOffset(compKey, mode) {{
  if (!placedModes[compKey]) placedModes[compKey] = [];
  const existing = placedModes[compKey];
  // Already placed this mode? Return null to skip
  if (existing.some(m => m.mode === mode)) return null;
  const offset = existing.length * 11;
  existing.push({{ mode }});
  return offset;
}}

const pathGroups = [];
DATA.paths.forEach((path, pi) => {{
  const color = path.color || "#c0392b";
  const op = Math.min(0.8, Math.max(0.35, Math.abs(path.score)/20));
  const sw = 2;
  const safe = color.replace("#","c");
  const hops = [...path.hops].reverse();

  const g = mkSvg("g", {{ class:"path-group", "data-idx":pi }});
  svg.appendChild(g);
  pathGroups.push(g);

  for (let i = 0; i < hops.length-1; i++) {{
    const lo = hops[i], hi = hops[i+1];
    const kLo = lo.component+"@"+lo.position, kHi = hi.component+"@"+hi.position;
    const iLo = compInfo[kLo], iHi = compInfo[kHi];
    if (!iLo || !iHi) continue;

    const loOff = streamOff(lo.position, pi);
    const hiOff = streamOff(hi.position, pi);
    const loOutY = iLo.isEmb ? iLo.cy-BOX_H/2 : iLo.outY;
    const hiInY = iHi.isEmb ? iHi.cy+BOX_H/2 : iHi.inY;
    const a = {{ stroke:color, "stroke-width":sw, "stroke-opacity":op }};

    // Write-out: curved from component top to stream
    if (!iLo.isEmb) {{
      g.appendChild(mkSvg("path", {{ class:"path-seg",
        d:`M${{iLo.cx+loOff}},${{iLo.cy-BOX_H/2}} Q${{iLo.sx+loOff}},${{iLo.cy-BOX_H/2}} ${{iLo.sx+loOff}},${{loOutY}}`,
        ...a }}));
    }}

    if (lo.position === hi.position) {{
      // Same position: up the stream
      g.appendChild(mkSvg("line", {{ class:"path-seg",
        x1:iLo.sx+loOff, y1:loOutY, x2:iHi.sx+hiOff, y2:hiInY, ...a }}));
      // Read-in: curved from stream to component bottom
      if (!iHi.isEmb) {{
        g.appendChild(mkSvg("path", {{ class:"path-seg",
          d:`M${{iHi.sx+hiOff}},${{hiInY}} Q${{iHi.sx+hiOff}},${{iHi.cy+BOX_H/2}} ${{iHi.cx+hiOff}},${{iHi.cy+BOX_H/2}}`,
          ...a, "marker-end":"url(#arr-"+safe+")" }}));
      }}
      // Mode tag (deduplicated)
      if (hi.mode) {{
        const moff = getModeOffset(kHi, hi.mode);
        if (moff !== null) {{
          g.appendChild(mkText(iHi.cx-BOX_W/2-4, iHi.cy-BOX_H/2+4+moff, hi.mode,
            {{ "text-anchor":"end", fill:color, "font-size":"8px", "font-weight":"700",
               opacity:0.7, class:"mode-label" }}));
        }}
      }}
    }} else {{
      // Cross-position via ghost
      const gk = hi.component+"@"+lo.position+"_ghost";
      const ghost = ghosts[gk];
      const tk = hi.component+"@"+hi.position;
      const glOff = ghostLinkOff(tk, pi);
      if (ghost) {{
        // Up stream to ghost
        g.appendChild(mkSvg("line", {{ class:"path-seg",
          x1:iLo.sx+loOff, y1:loOutY, x2:ghost.sx+loOff, y2:ghost.inY, ...a }}));
        // Curved into ghost
        g.appendChild(mkSvg("path", {{ class:"path-seg",
          d:`M${{ghost.sx+loOff}},${{ghost.inY}} Q${{ghost.sx+loOff}},${{ghost.cy+BOX_H/2}} ${{ghost.cx+loOff}},${{ghost.cy+BOX_H/2}}`,
          ...a }}));
        // Ghost → real (horizontal with offset)
        g.appendChild(mkSvg("line", {{ class:"path-seg ghost-link",
          x1:ghost.cx+BOX_W/2, y1:ghost.cy+glOff, x2:iHi.cx-BOX_W/2, y2:iHi.cy+glOff,
          ...a, "marker-end":"url(#arr-"+safe+")" }}));
        // Mode label (deduplicated)
        if (hi.mode) {{
          const crossKey = hi.component+"@"+hi.position+"_from_"+lo.position;
          const moff = getModeOffset(crossKey, hi.mode);
          if (moff !== null) {{
            const mx = (ghost.cx+BOX_W/2+iHi.cx-BOX_W/2)/2;
            g.appendChild(mkText(mx, ghost.cy-BOX_H/2-4-moff, hi.mode,
              {{ "text-anchor":"middle", fill:color, "font-size":"8px", "font-weight":"700",
                 opacity:0.7, class:"mode-label" }}));
          }}
        }}
      }}
    }}
  }}
}});

// ── Rerooted: draw connections from non-root top components up to root ──
if (DATA.root && rootInfo) {{
  const topAtRoot = topCompsByPos[rootInfo.pos] || [];
  const nFeed = topAtRoot.filter(e => e.comp !== DATA.root).length;
  let feedIdx = 0;
  topAtRoot.forEach(entry => {{
    if (entry.comp === DATA.root) return; // skip root itself
    const info = compInfo[entry.key];
    if (!info) return;
    const color = DATA.paths[entry.pi]?.color || "#c0392b";
    const op = 0.5;
    const off = (feedIdx - (nFeed - 1) / 2) * LANE_GAP;
    feedIdx++;
    // Component write-out to stream
    svg.appendChild(mkSvg("path", {{
      d: `M${{info.cx}},${{info.cy - BOX_H/2}} Q${{info.sx + off}},${{info.cy - BOX_H/2}} ${{info.sx + off}},${{info.outY}}`,
      fill:"none", stroke:color, "stroke-width":"2", "stroke-opacity":op
    }}));
    // Up stream to root's inY
    svg.appendChild(mkSvg("line", {{
      x1:info.sx + off, y1:info.outY, x2:rootInfo.sx + off, y2:rootInfo.inY,
      stroke:color, "stroke-width":"2", "stroke-opacity":op
    }}));
    // Stream to root component
    svg.appendChild(mkSvg("path", {{
      d: `M${{rootInfo.sx + off}},${{rootInfo.inY}} Q${{rootInfo.sx + off}},${{rootInfo.cy + BOX_H/2}} ${{rootInfo.cx}},${{rootInfo.cy + BOX_H/2}}`,
      fill:"none", stroke:color, "stroke-width":"2", "stroke-opacity":op,
      "marker-end": "url(#arr-" + color.replace("#","c") + ")"
    }}));
  }});
}}

// ── Legend ──
const active = new Set(DATA.paths.map((_,i)=>i));
const legendItems = [];
const legendBg = mkSvg("rect", {{ x:MARGIN.left-10, y:4, width:nCols*220, height:28,
  rx:6, fill:"#fff", stroke:"#eee", "stroke-width":0.5 }});
svg.appendChild(legendBg);

DATA.paths.forEach((path, i) => {{
  const lx = MARGIN.left + i * 210;
  if (lx + 200 > W) return;
  const color = path.color || "#c0392b";
  const label = (path.label||"path "+(i+1));
  const score = path.score ? " " + (path.score>0?"+":"") + path.score.toFixed(1) + "%" : "";

  const g = mkSvg("g", {{ class:"legend-item", "data-idx":i, transform:`translate(${{lx}},18)` }});
  g.appendChild(mkSvg("line", {{ x1:0, y1:0, x2:14, y2:0, stroke:color, "stroke-width":2.5, "stroke-linecap":"round", class:"legend-line" }}));
  g.appendChild(mkText(20, 4, label + score, {{ fill:"#666", "font-size":"8.5px", class:"legend-text", opacity:0.8 }}));
  svg.appendChild(g);
  legendItems.push(g);

  g.addEventListener("click", () => {{
    if (active.has(i)) active.delete(i); else active.add(i);
    updateVis();
  }});
  g.addEventListener("mouseenter", () => {{
    pathGroups.forEach((pg,j) => pg.classList.toggle("dimmed", j!==i));
    legendItems.forEach((li,j) => li.classList.toggle("dimmed", j!==i));
  }});
  g.addEventListener("mouseleave", () => updateVis());
}});

function updateVis() {{
  pathGroups.forEach((pg,i) => pg.classList.toggle("dimmed", !active.has(i)));
  legendItems.forEach((li,i) => li.classList.toggle("dimmed", !active.has(i)));
}}
</script></body></html>"""