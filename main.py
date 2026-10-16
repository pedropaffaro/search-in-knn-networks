from collections import deque
import heapq
import random
from math import hypot
from statistics import mean, stdev
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# ------------------------------ Geração da Rede KNN ------------------------------
def generate_vertices(n):
    """Gera n vértices com coordenadas (x, y) aleatórias no plano [0, n]."""
    return [
        {'x': random.randint(0, n), 'y': random.randint(0, n)}
        for _ in range(n)
    ]


def generate_edges_knn(vertices, k):
    """Gera arestas conectando cada vértice aos seus k vizinhos mais próximos."""
    edges_set = set()
    n = len(vertices)

    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dx = vertices[j]['x'] - vertices[i]['x']
                dy = vertices[j]['y'] - vertices[i]['y']
                dist = hypot(dx, dy)
                distances.append((dist, j))

        nearest_neighbors = heapq.nsmallest(k, distances, key=lambda x: x[0])

        for dist, j in nearest_neighbors:
            u, v = (i, j) if i < j else (j, i)
            edges_set.add((u, v, dist))

    return list(edges_set)


def generate_knn_network(n, k):
    vertices = generate_vertices(n)
    edges = generate_edges_knn(vertices, k)
    return vertices, edges


def build_adjacency_list(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v, weight in edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))
    return adj


def calculate_path_distance(path, adj):
    """Calcula a distância geométrica total percorrida em um caminho."""
    if not path or len(path) < 2:
        return 0.0
    total_dist = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for neighbor, w in adj[u]:
            if neighbor == v:
                total_dist += w
                break
    return total_dist


def heuristic(node, target, vertices):
    """Calcula a heurística h(n): distância euclidiana em linha reta até o destino."""
    dx = vertices[target]['x'] - vertices[node]['x']
    dy = vertices[target]['y'] - vertices[node]['y']
    return hypot(dx, dy)


# ------------------------------ Algoritmos de Busca ------------------------------
def dfs(adj, start, target):
    stack = [(start, [start])]
    visited = set()
    visited_order = []

    while stack:
        current_node, path = stack.pop()
        if current_node in visited:
            continue
        visited.add(current_node)
        visited_order.append(current_node)

        if current_node == target:
            return (
                path,
                calculate_path_distance(path, adj),
                len(visited_order),
                visited_order,
            )

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, float('inf'), len(visited_order), visited_order


def bfs(adj, start, target):
    queue = deque([(start, [start])])
    visited = {start}
    visited_order = []

    while queue:
        current_node, path = queue.popleft()
        visited_order.append(current_node)

        if current_node == target:
            return (
                path,
                calculate_path_distance(path, adj),
                len(visited_order),
                visited_order,
            )

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, float('inf'), len(visited_order), visited_order


def greedy_best_first(adj, vertices, start, target):
    start_h = heuristic(start, target, vertices)
    priority_queue = [(start_h, start, [start])]
    visited = set()
    visited_order = []

    while priority_queue:
        _, current_node, path = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        visited_order.append(current_node)

        if current_node == target:
            return (
                path,
                calculate_path_distance(path, adj),
                len(visited_order),
                visited_order,
            )

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                h_neighbor = heuristic(neighbor, target, vertices)
                heapq.heappush(
                    priority_queue, (h_neighbor, neighbor, path + [neighbor])
                )

    return None, float('inf'), len(visited_order), visited_order


def hill_climbing(adj, vertices, start, target):
    current_node = start
    path = [current_node]
    visited_order = [current_node]

    while True:
        if current_node == target:
            return (
                path,
                calculate_path_distance(path, adj),
                len(visited_order),
                visited_order,
            )

        current_h = heuristic(current_node, target, vertices)
        best_neighbor = None
        best_h = current_h

        for neighbor, _ in adj[current_node]:
            h_neighbor = heuristic(neighbor, target, vertices)
            if h_neighbor < best_h:
                best_h = h_neighbor
                best_neighbor = neighbor

        if best_neighbor is None:
            # Preso em mínimo local
            return None, float('inf'), len(visited_order), visited_order

        current_node = best_neighbor
        path.append(current_node)
        visited_order.append(current_node)


def dijkstra(adj, start, target):
    distances = {node: float('inf') for node in adj}
    distances[start] = 0.0
    priority_queue = [(0.0, start, [start])]
    visited_order = []

    while priority_queue:
        current_dist, current_node, path = heapq.heappop(priority_queue)
        if current_dist > distances[current_node]:
            continue

        visited_order.append(current_node)

        if current_node == target:
            return path, current_dist, len(visited_order), visited_order

        for neighbor, weight in adj[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(
                    priority_queue, (distance, neighbor, path + [neighbor])
                )

    return None, float('inf'), len(visited_order), visited_order


def a_star(adj, vertices, start, target):
    g_scores = {node: float('inf') for node in adj}
    g_scores[start] = 0.0
    h_start = heuristic(start, target, vertices)
    priority_queue = [(h_start, start, 0.0, [start])]
    visited_order = []

    while priority_queue:
        _, current_node, g_score, path = heapq.heappop(priority_queue)
        if g_score > g_scores[current_node]:
            continue

        visited_order.append(current_node)

        if current_node == target:
            return path, g_score, len(visited_order), visited_order

        for neighbor, weight in adj[current_node]:
            tentative_g = g_score + weight
            if tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_neighbor = tentative_g + heuristic(
                    neighbor, target, vertices
                )
                heapq.heappush(
                    priority_queue,
                    (f_neighbor, neighbor, tentative_g, path + [neighbor]),
                )

    return None, float('inf'), len(visited_order), visited_order


# ------------------------------ Experimento 1 ------------------------------
def animate_search(
    vertices, edges, visited_order, path, algo_name, filename
):
    if not visited_order:
        print(f'[{algo_name}] Sem passos registrados para animar.')
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Arestas do grafo no fundo
    for u, v, _ in edges:
        ax.plot(
            [vertices[u]['x'], vertices[v]['x']],
            [vertices[u]['y'], vertices[v]['y']],
            color='gainsboro',
            linewidth=0.8,
            zorder=1,
        )

    x_vals = [v['x'] for v in vertices]
    y_vals = [v['y'] for v in vertices]
    ax.scatter(
        x_vals, y_vals, color='lightgray', edgecolor='black', s=50, zorder=2
    )

    visited_scatter = ax.scatter(
        [],
        [],
        color='orange',
        edgecolor='darkorange',
        s=70,
        zorder=3,
        label='Explorado',
    )
    (solution_line,) = ax.plot(
        [],
        [],
        color='red',
        linewidth=3.0,
        marker='o',
        markersize=5,
        zorder=4,
        label='Caminho Solução',
    )

    start_node = visited_order[0]
    ax.scatter(
        [vertices[start_node]['x']],
        [vertices[start_node]['y']],
        color='green',
        s=130,
        zorder=5,
        label='Origem',
    )

    ax.set_title(f'Exploração Espaço de Busca - {algo_name}', fontsize=12)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)

    total_visited = len(visited_order)
    extra_frames_at_end = 15

    def update(frame):
        if frame < total_visited:
            nodes_so_far = visited_order[: frame + 1]
            xs = [vertices[node]['x'] for node in nodes_so_far]
            ys = [vertices[node]['y'] for node in nodes_so_far]
            visited_scatter.set_offsets(list(zip(xs, ys)))
        else:
            if path:
                px = [vertices[node]['x'] for node in path]
                py = [vertices[node]['y'] for node in path]
                solution_line.set_data(px, py)
        return visited_scatter, solution_line

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_visited + extra_frames_at_end,
        interval=40,
        blit=True,
    )

    print(f'-> Salvando animação "{filename}"...')
    anim.save(filename, writer='pillow', fps=20)
    
    img_filename = filename.replace('.gif', '_final.png')
    print(f'-> Salvando frame final "{img_filename}"...')
    
    xs = [vertices[node]['x'] for node in visited_order]
    ys = [vertices[node]['y'] for node in visited_order]
    visited_scatter.set_offsets(list(zip(xs, ys)))
    
    if path:
        px = [vertices[node]['x'] for node in path]
        py = [vertices[node]['y'] for node in path]
        solution_line.set_data(px, py)
        
    fig.savefig(img_filename, dpi=150, bbox_inches='tight')

    plt.close(fig)


def run_experiment_i():
    print('===============================================================')
    print(' Experimento (i): Animação da Busca em Rede KNN (N=150, k=5)')
    print('===============================================================')

    n_viz = 150
    k_viz = 5
    vertices, edges = generate_knn_network(n_viz, k_viz)
    adj = build_adjacency_list(n_viz, edges)

    start, target = 0, n_viz - 1

    algorithms = {
        'BFS': lambda: bfs(adj, start, target),
        'DFS': lambda: dfs(adj, start, target),
        'Greedy Best-First': lambda: greedy_best_first(
            adj, vertices, start, target
        ),
        'Hill Climbing': lambda: hill_climbing(adj, vertices, start, target),
        'Dijkstra': lambda: dijkstra(adj, start, target),
        'A*': lambda: a_star(adj, vertices, start, target),
    }

    for name, algo_fn in algorithms.items():
        path, dist, nodes, visited_order = algo_fn()
        filename = f"animations/busca_{name.replace(' ', '_').replace('*', 'Star')}.gif"

        if path:
            print(
                f'[{name:<17}] Distância: {dist:7.2f} | Saltos: {len(path)-1:3d} | Nós Expandidos: {nodes:3d}'
            )
            animate_search(
                vertices, edges, visited_order, path, name, filename
            )
        else:
            print(f'[{name:<17}] Caminho NÃO encontrado (mínimo local).')


# ------------------------------ Experimento 2 ------------------------------
def run_experiment_ii():
    print('\n===============================================================')
    print(' Experimento (ii): Testes Estatísticos em Redes KNN (N=2500)')
    print('===============================================================')

    n = 2500
    k_values = [3, 7, 11]
    num_pairs = 10

    random.seed(1)
    for k in k_values:

        random.seed(1)
        print(f'\n---> Configuração: (N = {n}, k = {k}) <---')
        vertices, edges = generate_knn_network(n, k)
        adj = build_adjacency_list(n, edges)

        pairs = []
        while len(pairs) < num_pairs:
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v and (u, v) not in pairs:
                pairs.append((u, v))

        algos = ['BFS', 'DFS', 'Greedy', 'HillClimbing', 'Dijkstra', 'A*']
        results = {algo: {'dist': [], 'time': []} for algo in algos}

        for start, target in pairs:
            for name, fn in [
                ('BFS', lambda: bfs(adj, start, target)),
                ('DFS', lambda: dfs(adj, start, target)),
                (
                    'Greedy',
                    lambda: greedy_best_first(adj, vertices, start, target),
                ),
                (
                    'HillClimbing',
                    lambda: hill_climbing(adj, vertices, start, target),
                ),
                ('Dijkstra', lambda: dijkstra(adj, start, target)),
                ('A*', lambda: a_star(adj, vertices, start, target)),
            ]:
                t0 = time.perf_counter()
                p, d, _, _ = fn()
                t1 = time.perf_counter()

                if p:
                    results[name]['dist'].append(d)
                    results[name]['time'].append(t1 - t0)

        print(
            f"{'Algoritmo':<15} | {'Dist Média':<12} | {'Tempo Médio (s)':<16} | {'Desvio Padrão Tempo':<18}"
        )
        print('-' * 70)
        for algo in algos:
            dists = results[algo]['dist']
            times = results[algo]['time']
            if dists:
                m_dist = mean(dists)
                m_time = mean(times)
                std_time = stdev(times) if len(times) > 1 else 0.0
                print(
                    f'{algo:<15} | {m_dist:<12.2f} | {m_time:<16.6f} | {std_time:<18.6f}'
                )
            else:
                print(f'{algo:<15} | Sem caminhos encontrados')


# ------------------------------ Experimento 3 ------------------------------
def run_experiment_iii():
    print('\n===============================================================')
    print(' Experimento (iii): Comparativo A* vs. Dijkstra (N=2500, k=7)')
    print('===============================================================')

    n = 2500
    k = 7
    vertices, edges = generate_knn_network(n, k)
    adj = build_adjacency_list(n, edges)

    pairs = []
    while len(pairs) < 10:
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if u != v and (u, v) not in pairs:
            pairs.append((u, v))

    print(
        f"{'Par (u, v)':<12} | {'Algoritmo':<10} | {'Distância':<10} | {'Nós Expandidos':<15} | {'Tempo (ms)':<10}"
    )
    print('-' * 67)

    for u, v in pairs:
        # Dijkstra
        t0 = time.perf_counter()
        p_dijk, d_dijk, nodes_dijk, _ = dijkstra(adj, u, v)
        t_dijk = (time.perf_counter() - t0) * 1000

        # A*
        t0 = time.perf_counter()
        p_astar, d_astar, nodes_astar, _ = a_star(adj, vertices, u, v)
        t_astar = (time.perf_counter() - t0) * 1000

        if p_dijk and p_astar:
            print(
                f'({u:<4}, {v:<4}) | Dijkstra   | {d_dijk:<10.2f} | {nodes_dijk:<15d} | {t_dijk:<10.3f}'
            )
            print(
                f'({'':<10}) | A*         | {d_astar:<10.2f} | {nodes_astar:<15d} | {t_astar:<10.3f}'
            )
            print('-' * 67)


# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    random.seed(1)
    run_experiment_i()
    run_experiment_ii()
    run_experiment_iii()