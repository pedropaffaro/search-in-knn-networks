import heapq
import random
import time
from collections import deque
from math import exp, hypot
from statistics import mean, stdev
import matplotlib.pyplot as plt


# --------------------- Geração da Rede Geográfica --------------------- #
def generate_vertices(n):
    return [
        {'x': random.randint(0, n), 'y': random.randint(0, n)}
        for _ in range(n)
    ]


def generate_edges(vertices, lam):
    edges = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            dx = vertices[j]['x'] - vertices[i]['x']
            dy = vertices[j]['y'] - vertices[i]['y']
            s_ij = hypot(dx, dy)
            p = exp(-lam * s_ij)
            if random.random() < p:
                edges.append((i, j, s_ij))
    return edges


def generate_geographic_network(n, lam):
    vertices = generate_vertices(n)
    edges = generate_edges(vertices, lam)
    return vertices, edges


def build_adjacency_list(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v, weight in edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))
    return adj


def calculate_path_distance(path, adj):
    """Calcula a distância geográfica total percorrida num caminho."""
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
    """Distância euclidiana em linha reta h(n)."""
    dx = vertices[target]['x'] - vertices[node]['x']
    dy = vertices[target]['y'] - vertices[node]['y']
    return hypot(dx, dy)


# --------------------- Algoritmos de Busca Direcionados --------------------- #
def dfs(adj, start, target):
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0

    while stack:
        current_node, path = stack.pop()
        if current_node in visited:
            continue
        visited.add(current_node)
        nodes_expanded += 1

        if current_node == target:
            return path, calculate_path_distance(path, adj), nodes_expanded

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, float('inf'), nodes_expanded


def bfs(adj, start, target):
    queue = deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current_node, path = queue.popleft()
        nodes_expanded += 1

        if current_node == target:
            return path, calculate_path_distance(path, adj), nodes_expanded

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, float('inf'), nodes_expanded


def greedy_best_first(adj, vertices, start, target):
    start_h = heuristic(start, target, vertices)
    priority_queue = [(start_h, start, [start])]
    visited = set()
    nodes_expanded = 0

    while priority_queue:
        _, current_node, path = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        nodes_expanded += 1

        if current_node == target:
            return path, calculate_path_distance(path, adj), nodes_expanded

        for neighbor, _ in adj[current_node]:
            if neighbor not in visited:
                h_neighbor = heuristic(neighbor, target, vertices)
                heapq.heappush(
                    priority_queue, (h_neighbor, neighbor, path + [neighbor])
                )

    return None, float('inf'), nodes_expanded


def hill_climbing(adj, vertices, start, target):
    current_node = start
    path = [current_node]
    nodes_expanded = 0

    while True:
        nodes_expanded += 1
        if current_node == target:
            return path, calculate_path_distance(path, adj), nodes_expanded

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
            return None, float('inf'), nodes_expanded

        current_node = best_neighbor
        path.append(current_node)


def dijkstra(adj, start, target):
    distances = {node: float('inf') for node in adj}
    distances[start] = 0.0
    priority_queue = [(0.0, start, [start])]
    nodes_expanded = 0

    while priority_queue:
        current_dist, current_node, path = heapq.heappop(priority_queue)
        if current_dist > distances[current_node]:
            continue

        nodes_expanded += 1
        if current_node == target:
            return path, current_dist, nodes_expanded

        for neighbor, weight in adj[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(
                    priority_queue, (distance, neighbor, path + [neighbor])
                )

    return None, float('inf'), nodes_expanded


def a_star(adj, vertices, start, target):
    g_scores = {node: float('inf') for node in adj}
    g_scores[start] = 0.0
    h_start = heuristic(start, target, vertices)
    priority_queue = [(h_start, start, 0.0, [start])]
    nodes_expanded = 0

    while priority_queue:
        _, current_node, g_score, path = heapq.heappop(priority_queue)
        if g_score > g_scores[current_node]:
            continue

        nodes_expanded += 1
        if current_node == target:
            return path, g_score, nodes_expanded

        for neighbor, weight in adj[current_node]:
            tentative_g = g_score + weight
            if tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_neighbor = tentative_g + heuristic(neighbor, target, vertices)
                heapq.heappush(
                    priority_queue,
                    (f_neighbor, neighbor, tentative_g, path + [neighbor]),
                )

    return None, float('inf'), nodes_expanded


# --------------------- Experimento (i): Visualização --------------------- #
def plot_experiment_i(vertices, edges, path, title='Visualização da Busca'):
    plt.figure(figsize=(9, 7))

    # Desenhar arestas
    for u, v, _ in edges:
        plt.plot(
            [vertices[u]['x'], vertices[v]['x']],
            [vertices[u]['y'], vertices[v]['y']],
            color='gainsboro',
            linewidth=0.8,
            zorder=1,
        )

    # Desenhar vértices
    x_vals = [v['x'] for v in vertices]
    y_vals = [v['y'] for v in vertices]
    plt.scatter(
        x_vals,
        y_vals,
        color='skyblue',
        edgecolor='black',
        s=60,
        zorder=2,
        label='Vértices',
    )

    # Destacar o caminho percorrido
    if path:
        path_x = [vertices[node]['x'] for node in path]
        path_y = [vertices[node]['y'] for node in path]
        plt.plot(
            path_x,
            path_y,
            color='red',
            linewidth=2.5,
            marker='o',
            markersize=5,
            zorder=3,
            label='Caminho Solução',
        )
        plt.scatter(
            [path_x[0]],
            [path_y[0]],
            color='green',
            s=120,
            zorder=4,
            label='Origem',
        )
        plt.scatter(
            [path_x[-1]],
            [path_y[-1]],
            color='purple',
            s=120,
            zorder=4,
            label='Destino',
        )

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def run_experiment_i():
    print('=== Experimento (i): Visualização em Rede Pequena (N=150) ===')
    n_viz = 150
    lam_viz = 0.03
    vertices, edges = generate_geographic_network(n_viz, lam_viz)
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
        path, dist, _ = algo_fn()
        if path:
            print(
                f'[{name}] Distância: {dist:.2f} | Saltos: {len(path)-1} | Caminho: {path}'
            )
            plot_experiment_i(
                vertices, edges, path, title=f'Experimento i - Algoritmo: {name}'
            )
        else:
            print(f'[{name}] Caminho não encontrado (mínimo local ou desconexo).')


# --------------------- Experimento (ii): Testes Estatísticos --------------------- #
def run_experiment_ii():
    print(
        '\n=== Experimento (ii): Avaliação Estatística em Redes Geográficas (N=2500) ==='
    )
    n = 2500
    lambdas = [0.01, 0.02, 0.03]
    num_pairs = 10

    for lam in lambdas:
        print(f'\n--- Configuração: (N = {n}, λ = {lam}) ---')
        vertices, edges = generate_geographic_network(n, lam)
        adj = build_adjacency_list(n, edges)

        # Selecionar 10 pares distintos de origem e destino
        pairs = []
        while len(pairs) < num_pairs:
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v and (u, v) not in pairs:
                pairs.append((u, v))

        algos = ['BFS', 'DFS', 'Greedy', 'HillClimbing', 'Dijkstra', 'A*']
        results = {
            algo: {'dist': [], 'time': []} for algo in algos
        }

        for start, target in pairs:
            # BFS
            t0 = time.perf_counter()
            p, d, _ = bfs(adj, start, target)
            t1 = time.perf_counter()
            if p:
                results['BFS']['dist'].append(d)
                results['BFS']['time'].append(t1 - t0)

            # DFS
            t0 = time.perf_counter()
            p, d, _ = dfs(adj, start, target)
            t1 = time.perf_counter()
            if p:
                results['DFS']['dist'].append(d)
                results['DFS']['time'].append(t1 - t0)

            # Greedy Best-First
            t0 = time.perf_counter()
            p, d, _ = greedy_best_first(adj, vertices, start, target)
            t1 = time.perf_counter()
            if p:
                results['Greedy']['dist'].append(d)
                results['Greedy']['time'].append(t1 - t0)

            # Hill Climbing
            t0 = time.perf_counter()
            p, d, _ = hill_climbing(adj, vertices, start, target)
            t1 = time.perf_counter()
            if p:
                results['HillClimbing']['dist'].append(d)
                results['HillClimbing']['time'].append(t1 - t0)

            # Dijkstra
            t0 = time.perf_counter()
            p, d, _ = dijkstra(adj, start, target)
            t1 = time.perf_counter()
            if p:
                results['Dijkstra']['dist'].append(d)
                results['Dijkstra']['time'].append(t1 - t0)

            # A*
            t0 = time.perf_counter()
            p, d, _ = a_star(adj, vertices, start, target)
            t1 = time.perf_counter()
            if p:
                results['A*']['dist'].append(d)
                results['A*']['time'].append(t1 - t0)

        # Imprimir resultados consolidados
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


# --------------------- Experimento (iii): A* vs. Dijkstra --------------------- #
def run_experiment_iii():
    print(
        '\n=== Experimento (iii): Comparativo Direto A* vs. Dijkstra (N=2500, λ=0.01) ==='
    )
    n = 2500
    lam = 0.01
    vertices, edges = generate_geographic_network(n, lam)
    adj = build_adjacency_list(n, edges)

    pairs = []
    while len(pairs) < 10:
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if u != v:
            pairs.append((u, v))

    print(
        f"{'Par (u, v)':<12} | {'Algoritmo':<10} | {'Distância':<10} | {'Nós Expandidos':<15} | {'Tempo (ms)':<10}"
    )
    print('-' * 65)

    for u, v in pairs:
        # Dijkstra
        t0 = time.perf_counter()
        p_dijk, d_dijk, nodes_dijk = dijkstra(adj, u, v)
        t_dijk = (time.perf_counter() - t0) * 1000

        # A*
        t0 = time.perf_counter()
        p_astar, d_astar, nodes_astar = a_star(adj, vertices, u, v)
        t_astar = (time.perf_counter() - t0) * 1000

        if p_dijk and p_astar:
            print(
                f'({u:<4}, {v:<4}) | Dijkstra   | {d_dijk:<10.2f} | {nodes_dijk:<15} | {t_dijk:<10.3f}'
            )
            print(
                f'({'':<10}) | A*         | {d_astar:<10.2f} | {nodes_astar:<15} | {t_astar:<10.3f}'
            )
            print('-' * 65)


if __name__ == '__main__':
    run_experiment_i()
    run_experiment_ii()
    run_experiment_iii()