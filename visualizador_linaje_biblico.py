"""
Visualizador HTML del linaje bíblico (Génesis -> Apocalipsis).

Lee un CSV con columnas:
- name
- father
- mother
- spouse (opcional)

Genera un HTML interactivo con layout jerárquico, colores por tipo de relación,
estadísticas y controles de navegación.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import networkx as nx
import pandas as pd
from pyvis.network import Network


@dataclass
class Config:
    input_csv: Path
    output_html: Path
    max_nodes: int | None
    include_spouses: bool


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    if text.startswith("http://www.wikidata.org/.well-known"):
        return ""
    return text


def load_data(csv_path: Path) -> pd.DataFrame:
    """Carga y normaliza el CSV en UTF-8 o latin-1."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    required = {"name", "father", "mother"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    for column in ["name", "father", "mother", "spouse"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].map(_clean)

    # Quita registros sin persona.
    df = df[df["name"] != ""].copy()
    return df


def build_graph(df: pd.DataFrame, include_spouses: bool = True) -> nx.DiGraph:
    """Construye un grafo dirigido parent -> child y opcionalmente spouse."""
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        person = row["name"]
        father = row["father"]
        mother = row["mother"]
        spouse = row.get("spouse", "")

        graph.add_node(person)

        if father:
            graph.add_node(father)
            graph.add_edge(father, person, relation="father", color="#2E86DE", width=2)

        if mother:
            graph.add_node(mother)
            graph.add_edge(mother, person, relation="mother", color="#E84393", width=2)

        if include_spouses and spouse:
            graph.add_node(spouse)
            # Arista doble para que ambos aparezcan conectados en modo dirigido.
            graph.add_edge(person, spouse, relation="spouse", color="#95A5A6", width=1, dashes=True)
            graph.add_edge(spouse, person, relation="spouse", color="#95A5A6", width=1, dashes=True)

    return graph


def generation_levels(graph: nx.DiGraph) -> Dict[str, int]:
    """Calcula nivel generacional aproximado usando distancia máxima desde raíces."""
    if graph.number_of_nodes() == 0:
        return {}

    parent_graph = nx.DiGraph(
        (u, v) for u, v, attrs in graph.edges(data=True) if attrs.get("relation") in {"father", "mother"}
    )

    levels: Dict[str, int] = {node: 0 for node in graph.nodes}
    if parent_graph.number_of_nodes() == 0:
        return levels

    try:
        order = list(nx.topological_sort(parent_graph))
        for node in order:
            for child in parent_graph.successors(node):
                levels[child] = max(levels.get(child, 0), levels.get(node, 0) + 1)
    except nx.NetworkXUnfeasible:
        # Si hay ciclos/inconsistencias de datos, se preserva nivel 0.
        pass

    return levels


def descendants_count(graph: nx.DiGraph) -> Dict[str, int]:
    """Número de descendientes por nodo (solo enlaces padre/madre)."""
    parent_graph = nx.DiGraph(
        (u, v) for u, v, attrs in graph.edges(data=True) if attrs.get("relation") in {"father", "mother"}
    )
    counts: Dict[str, int] = {}
    for node in graph.nodes:
        if node in parent_graph:
            counts[node] = len(nx.descendants(parent_graph, node))
        else:
            counts[node] = 0
    return counts


def maybe_limit_graph(graph: nx.DiGraph, max_nodes: int | None) -> nx.DiGraph:
    """Limita el tamaño del grafo manteniendo nodos más conectados."""
    if not max_nodes or graph.number_of_nodes() <= max_nodes:
        return graph

    ranked = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    selected = {node for node, _ in ranked[:max_nodes]}
    return graph.subgraph(selected).copy()


def style_network(graph: nx.DiGraph) -> Network:
    """Convierte el grafo a PyVis con estilos legibles para datasets grandes."""
    net = Network(
        height="95vh",
        width="100%",
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        directed=True,
        select_menu=True,
        filter_menu=True,
    )
    net.from_nx(graph)

    levels = generation_levels(graph)
    descendants = descendants_count(graph)

    for node in net.nodes:
        node_id = node["id"]
        level = levels.get(node_id, 0)
        desc = descendants.get(node_id, 0)

        size = 14 + min(desc, 25)
        node["size"] = size
        node["level"] = int(level)
        node["shape"] = "dot"
        node["color"] = {
            "background": "#22c55e" if level == 0 else "#38bdf8",
            "border": "#ffffff",
            "highlight": {"background": "#f59e0b", "border": "#ffffff"},
        }
        node["title"] = (
            f"<b>{node_id}</b><br>Generación aprox.: {level}<br>"
            f"Descendientes detectados: {desc}"
        )

    for edge in net.edges:
        relation = edge.get("relation", "")
        if relation == "father":
            edge["label"] = "Padre"
            edge["arrows"] = "to"
        elif relation == "mother":
            edge["label"] = "Madre"
            edge["arrows"] = "to"
        else:
            edge["label"] = "Cónyuge"
            edge["arrows"] = "to"

    options = """
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 180,
          "treeSpacing": 220,
          "levelSeparation": 170,
          "blockShifting": true
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "cubicBezier",
          "roundness": 0.35
        },
        "font": {
          "size": 12,
          "align": "top"
        }
      },
      "nodes": {
        "font": {
          "size": 16,
          "strokeWidth": 2,
          "strokeColor": "#0f172a"
        }
      }
    }
    """
    net.set_options(options)

    return net


def inject_legend(html: str, node_count: int, edge_count: int) -> str:
    legend = f"""
    <div style="position:fixed;top:12px;left:12px;z-index:9999;background:#111827;color:#e5e7eb;
                border:1px solid #374151;border-radius:10px;padding:12px 14px;font-family:Arial,sans-serif;
                max-width:320px;box-shadow:0 8px 20px rgba(0,0,0,.35)">
      <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Linaje Bíblico</div>
      <div style="font-size:13px;margin-bottom:4px;">Nodos: <b>{node_count}</b> · Relaciones: <b>{edge_count}</b></div>
      <div style="font-size:12px;line-height:1.45;">
        <span style="color:#2E86DE">■</span> Padre → Hijo<br>
        <span style="color:#E84393">■</span> Madre → Hijo<br>
        <span style="color:#95A5A6">■</span> Cónyuge ↔ Cónyuge<br>
        <span style="color:#22c55e">●</span> Raíces (sin padres conocidos)
      </div>
    </div>
    """

    marker = "<body>"
    if marker in html:
        return html.replace(marker, f"{marker}\n{legend}", 1)
    return legend + html


def generate_html(config: Config) -> None:
    df = load_data(config.input_csv)
    graph = build_graph(df, include_spouses=config.include_spouses)
    graph = maybe_limit_graph(graph, config.max_nodes)
    net = style_network(graph)

    html = net.generate_html()
    html = inject_legend(html, graph.number_of_nodes(), graph.number_of_edges())

    config.output_html.write_text(html, encoding="utf-8")
    print(f"✅ HTML generado: {config.output_html}")
    print(f"   Nodos: {graph.number_of_nodes()} | Aristas: {graph.number_of_edges()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Genera un HTML del linaje bíblico desde un CSV.")
    parser.add_argument("--input", default="biblical_genealogy_complete_real.csv", help="Ruta del CSV fuente")
    parser.add_argument("--output", default="linaje_biblico_interactivo.html", help="Ruta del HTML de salida")
    parser.add_argument("--max-nodes", type=int, default=None, help="Limita nodos para mejorar legibilidad")
    parser.add_argument(
        "--no-spouses",
        action="store_true",
        help="Desactiva relaciones de cónyuge para enfocarse en línea sanguínea",
    )
    args = parser.parse_args()

    return Config(
        input_csv=Path(args.input),
        output_html=Path(args.output),
        max_nodes=args.max_nodes,
        include_spouses=not args.no_spouses,
    )


if __name__ == "__main__":
    cfg = parse_args()
    generate_html(cfg)
