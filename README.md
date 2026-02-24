# GPT_BIBLIE_Graph

Visualizador interactivo del linaje de personajes bíblicos en HTML (desde Génesis hasta Apocalipsis) usando Python + NetworkX + PyVis.

## Script principal

- `visualizador_linaje_biblico.py`

## Requisitos

```bash
pip install pandas networkx pyvis
```

## Uso

```bash
python visualizador_linaje_biblico.py \
  --input biblical_genealogy_complete_real.csv \
  --output linaje_biblico_interactivo.html
```

### Opciones útiles

- `--max-nodes 400`: limita el número de nodos para una visualización más legible.
- `--no-spouses`: excluye relaciones de cónyuge y deja solo padre/madre → hijo.

## Qué mejora esta versión

- Layout jerárquico vertical por generaciones aproximadas.
- Diferenciación visual de padre, madre y cónyuge.
- Tamaño de nodo según cantidad de descendientes detectados.
- Panel flotante con leyenda y conteo de nodos/aristas.
- Menús de filtro y selección integrados en el HTML.
