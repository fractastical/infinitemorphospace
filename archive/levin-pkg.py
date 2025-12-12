import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
from pyvis.network import Network
import webbrowser

# --- Step 1: Extract and Structure Data ---
# This data is structured as RDF triples based on Michael Levin's publication information.

# Define namespaces for our ontology
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

# Triples representing the data about Michael Levin's work
# (Subject, Predicate, Object)
levin_triples =

# --- Step 2: Build a Semantic Graph with rdflib ---
g = Graph()
for s, p, o in levin_triples:
    g.add((s, p, o))

# --- Step 3: Convert to a networkx Graph ---
# This conversion allows us to use a wide range of graph algorithms and visualization tools.
nx_graph = rdflib_to_networkx_multidigraph(g)

# --- Step 4: Generate an Interactive Visualization with pyvis ---
# Create a Pyvis network object
# You can customize the size and appearance of the graph
net = Network(height='800px', width='100%', notebook=True, cdn_resources='remote', directed=True)

# Translate the networkx graph to a Pyvis graph
net.from_nx(nx_graph)

# Add custom styling and metadata for better visualization
for node in net.nodes:
    uri = URIRef(node["id"])
    # Use the 'name' literal as the display label
    try:
        label = g.value(subject=uri, predicate=SCHEMA.name)
        if label:
            node["label"] = str(label)
            node["title"] = str(label) # Tooltip on hover
    except:
        pass

    # Color nodes based on their type (a simple heuristic based on URI)
    if "Paper" in str(uri):
        node["color"] = "#f4a261" # Orange for papers
        node["shape"] = "box"
    elif "University" in str(uri) or "Center" in str(uri):
        node["color"] = "#2a9d8f" # Teal for institutions
    elif any(kw in str(uri) for kw in):
        node["color"] = "#e76f51" # Red for concepts
        node["shape"] = "ellipse"
    else:
        node["color"] = "#264653" # Dark blue for people

# Add physics layout buttons for interactive adjustments
net.show_buttons(filter_=['physics'])

# Generate the HTML file
file_name = "levin_knowledge_graph.html"
net.show(file_name)

print(f"Interactive knowledge graph has been generated: {file_name}")
# Open the generated file in the default web browser
webbrowser.open(file_name)