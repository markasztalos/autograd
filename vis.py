import os
import tempfile
from graphviz import Digraph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root, hide_without_labels=False):
    dot = Digraph(
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'bgcolor': '#0d1117',
            'splines': 'true',
            'pad': '0.5',
            'nodesep': '0.5',
            'ranksep': '0.7',
            'fontname': 'Helvetica',
        },
        edge_attr={
            'penwidth': '2',
        },
    )

    _build_graph(root, dot, {}, hide_without_labels)

    outdir = tempfile.mkdtemp()
    path = os.path.join(outdir, 'autograd_graph')
    dot.render(path, cleanup=True)
    return f"file://{path}.svg"


def _build_graph(node, dot, connections, hide_without_labels):
    if id(node) in connections:
        return connections[id(node)]
    connections[id(node)] = []

    node_id = str(id(node))
    hide = hide_without_labels and not node.label

    child_sources = []
    for child in node._prev:
        child_sources.extend(
            _build_graph(child, dot, connections, hide_without_labels)
        )

    if node._op:
        op_id = f"op_{id(node)}"
        dot.node(
            op_id,
            label=node._op,
            shape='circle',
            style='filled',
            fillcolor='#fbbf24',
            fontcolor='#1f2937',
            fontname='Helvetica Bold',
            fontsize='18',
            color='#f59e0b',
            penwidth='3',
            width='0.4',
            height='0.4',
        )
        for src in child_sources:
            dot.edge(src, op_id, color='#8b949e')

        if not hide:
            _add_value_node(dot, node, node_id)
            dot.edge(op_id, node_id, color='#8b949e')
            result = [node_id]
        else:
            result = [op_id]
    else:
        if not hide:
            _add_value_node(dot, node, node_id)
            result = [node_id]
        else:
            result = []

    connections[id(node)] = result
    return result


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def _add_value_node(dot, node, node_id):
    lines = [f"data = {_fmt(node.data)}"]
    if node.label:
        lines.insert(0, node.label)
    lines.append(f"grad = {_fmt(node.grad)}")
    for k, v in node._meta.items():
        lines.append(f"{k} = {_fmt(v)}")

    if node.is_param:
        fill = '#1a3a2e:#0d1117'
        fontcolor = '#3fb950'
        border = '#2ea043'
    else:
        fill = '#1c2333:#0d1117'
        fontcolor = '#79c0ff'
        border = '#30363d'

    dot.node(
        node_id,
        label='\n'.join(lines),
        shape='box',
        style='filled,rounded',
        fillcolor=fill,
        gradientangle='270',
        fontname='Helvetica',
        fontcolor=fontcolor,
        color=border,
        penwidth='2',
    )


def vis_2d(xs, ys, w, b):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    ax.scatter(xs, ys, color='#58a6ff', s=12, alpha=0.6, label='data')
    ax.plot(xs, [w * x + b for x in xs], color='#3fb950', linewidth=2, label='model')

    ax.set_xlabel('x', color='#8b949e')
    ax.set_ylabel('y', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.legend(facecolor='#161b22', labelcolor='#8b949e', edgecolor='#30363d')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    outdir = tempfile.mkdtemp()
    path = os.path.join(outdir, 'autograd_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return f"file://{path}"
