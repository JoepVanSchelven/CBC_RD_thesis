import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def create_G(df_lines: pd.DataFrame) -> nx.Graph:
    # Create a directed graph using NetworkX
    G = nx.Graph()

    # Add nodes (buses)
    from_buses = list(df_lines['from_bus'].values)
    to_buses = list(df_lines['to_bus'].values)

    for bus in set(from_buses + to_buses):
        G.add_node(bus)

    # Add edges (lines) with attributes
    for i in range(len(df_lines)):
        from_bus = from_buses[i]
        to_bus = to_buses[i]
        G.add_edge(from_bus, to_bus, capacity=df_lines['capacity'][i], susceptance=df_lines['susceptance'][i])

    return G


# Define a function to draw the network with node labels and edge attributes
def draw_network(df_lines: pd.DataFrame):
    G = create_G(df_lines)

    pos = nx.spring_layout(G)  # Positioning layout (e.g., spring layout)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Optionally, add edge labels (e.g., susceptance or capacity)
    edge_labels = {(i, j): f"b={G[i][j]['susceptance']}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Transmission Network")
    plt.axis('off')
    plt.show()


def draw_network_with_power_flows(df_lines, df_flows):
    G = create_G(df_lines)

    pos = nx.spring_layout(G)   # Positioning layout (e.g., spring layout)
    plt.figure(figsize=(14, 12))

    # Map power flows to edges
    power_flows = {}
    for idx, row in df_flows.iterrows():
        from_bus = df_lines.iloc[idx]['from_bus']
        to_bus = df_lines.iloc[idx]['to_bus']
        power_flows[(from_bus, to_bus)] = row['flow']  # Take absolute values for flow magnitudes

    # Color edges based on power flows (optional)
    edge_colors = [power_flows.get((i, j), power_flows.get((j, i))) for i, j in G.edges]  # Power flows dictionary

    # Draw the graph with edge colors based on power flows
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn_r)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Add edge labels (e.g., power flows)
    edge_labels = {(i, j): f"{power_flows.get((i, j), power_flows.get((j, i))):.2f} MW" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Add colorbar to the graph
    min_flow = min(power_flows.values())
    max_flow = max(power_flows.values())

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=min_flow, vmax=max_flow))
    sm.set_array([])  # Empty array for the colorbar
    plt.colorbar(sm, ax=plt.gca(), label="Power Flow (MW)")

    plt.title("Transmission Network with Power Flows")
    plt.axis('off')
    plt.show()


def draw_network_with_absolute_power_flows(df_lines, df_flows):
    G = create_G(df_lines)

    pos = nx.spring_layout(G)   # Positioning layout (e.g., spring layout)
    plt.figure(figsize=(14, 12))

    # Map power flows to edges
    power_flows = {}
    for idx, row in df_flows.iterrows():
        from_bus = df_lines.iloc[idx]['from_bus']
        to_bus = df_lines.iloc[idx]['to_bus']
        power_flows[(from_bus, to_bus)] = abs(row['flow'])  # Take absolute values for flow magnitudes

    # Color edges based on power flows (optional)
    edge_colors = [power_flows.get((i, j), power_flows.get((j, i))) for i, j in G.edges]  # Power flows dictionary

    # Draw the graph with edge colors based on power flows
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn_r)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Add edge labels (e.g., power flows)
    edge_labels = {(i, j): f"{power_flows.get((i, j), power_flows.get((j, i))):.2f} MW" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Add colorbar to the graph
    min_flow = min(power_flows.values())
    max_flow = max(power_flows.values())

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=min_flow, vmax=max_flow))
    sm.set_array([])  # Empty array for the colorbar
    plt.colorbar(sm, ax=plt.gca(), label="Absolute Power Flow (MW)")

    plt.title("Transmission Network with Power Flows")
    plt.axis('off')
    plt.show()


def draw_network_with_congestion(df_lines, df_flows):
    G = create_G(df_lines)

    pos = nx.spring_layout(G)   # Positioning layout (e.g., spring layout)
    plt.figure(figsize=(14, 12))

    # Map power flows to edges
    congestion = {}
    for idx, row in df_flows.iterrows():
        from_bus = df_lines.iloc[idx]['from_bus']
        to_bus = df_lines.iloc[idx]['to_bus']
        congestion[(from_bus, to_bus)] = max(0.0, abs(row['flow']) - df_lines.iloc[idx]['capacity'])

    # Color edges based on power flows (optional)
    edge_colors = [congestion.get((i, j), congestion.get((j, i))) for i, j in G.edges]

    # Draw the graph with edge colors based on power flows
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Add edge labels (e.g., power flows)
    edge_labels = {(i, j): f"{congestion.get((i, j), congestion.get((j, i))):.2f} MW" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Add colorbar to the graph
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])  # Empty array for the colorbar
    plt.colorbar(sm, ax=plt.gca(), label="Congestion")

    plt.title("Transmission Network with Power Flows")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    line_capacity = 100
    susceptance = 1.0e4
    n_lines = 8

    # Define your network
    line_data = {
        'from_bus': [1, 5, 2, 6, 3, 7, 4, 8],
        'to_bus': [5, 6, 6, 7, 7, 8, 8, 5],
        'capacity': [line_capacity] * n_lines,  # Replace with your actual line capacity
        'susceptance': [susceptance] * n_lines  # Replace with your actual susceptance values
    }

    flow_data = {'flow': [100.0, 5.0, 100.0, 105.0, -120.0, -15.0, -80.0, -95.0]}

    df_lines = pd.DataFrame(line_data)
    df_flow = pd.DataFrame(flow_data)

    draw_network(df_lines)

    draw_network_with_power_flows(df_lines, df_flow)

    draw_network_with_absolute_power_flows(df_lines, df_flow)

    draw_network_with_congestion(df_lines, df_flow)