# tsp_solver.py
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import time

try:
    from dwave.system import LeapHybridSampler
    QUANTUM_ENABLED = True
except ImportError:
    QUANTUM_ENABLED = False

class TSPSolver:
    def __init__(self, points, quantum_token=None):
        """
        Inicializa el solver con las coordenadas de las ciudades.
        
        Args:
            points (np.array): Matriz Nx2 con coordenadas [x, y] de las ciudades.
            quantum_token (str): API key para D-Wave Leap (opcional).
        """
        self.points = np.array(points)
        self.dist_matrix = distance_matrix(self.points, self.points)
        self.n_points = len(points)
        self.sampler = LeapHybridSampler(token=quantum_token) if (QUANTUM_ENABLED and quantum_token) else None
        self._setup_params()

    def _setup_params(self):
        """Configura parámetros por defecto."""
        self.params = {
            'clustering': {
                'threshold': 0.2,
                'linkage': 'complete'
            },
            'quantum': {
                'num_reads': 1000,
                'time_limit': 20
            },
            'classic': {
                '3opt_iters': 3,
                'hybrid_ratio': 0.6
            }
        }

    def solve(self, time_limit=300):
        """
        Resuelve el TSP con tiempo límite.
        
        Args:
            time_limit (int): Tiempo máximo en segundos.
            
        Returns:
            tuple: (ruta, distancia)
        """
        start_time = time.time()
        
        # 1. Clustering adaptativo
        clusters = self._cluster_points()
        time_left = time_limit - (time.time() - start_time)
        
        # 2. Asignación dinámica de tiempo
        quantum_time = time_left * self.params['classic']['hybrid_ratio']
        classic_time = time_left - quantum_time
        
        # 3. Resolver subproblemas en paralelo
        sub_tours = self._solve_subproblems(clusters, quantum_time)
        
        # 4. Ensamblar solución global
        global_tour = self._assemble_global_tour(clusters, sub_tours)
        
        # 5. Refinamiento clásico
        optimized_tour = self._classic_optimization(global_tour, classic_time)
        
        return optimized_tour, self._calculate_distance(optimized_tour)

    def _cluster_points(self):
        """Clustering jerárquico con umbral adaptativo."""
        norm_dist = self.dist_matrix / np.max(self.dist_matrix)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.params['clustering']['linkage'],
            distance_threshold=self.params['clustering']['threshold']
        )
        model.fit(norm_dist)
        return [self.points[model.labels_ == i] for i in np.unique(model.labels_)]

    def _solve_subproblems(self, clusters, time_limit):
        """Resuelve clusters en paralelo."""
        time_per_cluster = max(5, time_limit / len(clusters))
        sub_tours = []
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for cluster in clusters:
                futures.append(executor.submit(
                    self._solve_cluster,
                    cluster,
                    min(time_per_cluster, 30)  # Límite de 30s por cluster
                ))
            
            for future in futures:
                sub_tours.append(future.result())
        
        return sub_tours

    def _solve_cluster(self, cluster, time_limit):
        """Estrategia de solución por cluster."""
        if len(cluster) <= 8:
            return self._brute_force(cluster)
        
        if self.sampler and QUANTUM_ENABLED:
            try:
                return self._quantum_annealing(cluster, time_limit)
            except:
                return self._classic_heuristic(cluster)
        else:
            return self._classic_heuristic(cluster)

    def _quantum_annealing(self, points, time_limit):
        """Resolución con D-Wave."""
        qubo = self._create_qubo(points)
        response = self.sampler.sample(
            qubo,
            time_limit=time_limit,
            num_reads=self.params['quantum']['num_reads']
        )
        return points[list(response.first.sample.keys())[0][0]]

    def _create_qubo(self, points):
        """Genera modelo QUBO para el TSP."""
        n = len(points)
        dist = distance_matrix(points, points)
        max_dist = np.max(dist)
        Q = {}
        
        # Términos de costo
        for i in range(n):
            for j in range(n):
                if i != j:
                    for t in range(n):
                        weight = dist[i][j] / max_dist
                        Q[((i, t), (j, (t+1)%n)] = weight
        
        # Restricciones
        penalty = 2.0 * np.mean(dist)
        for v in range(n):
            for t1 in range(n):
                for t2 in range(t1+1, n):
                    Q[((v, t1), (v, t2))] = penalty
        
        return Q

    def _classic_heuristic(self, points):
        """Algoritmo clásico basado en convex hull + 2-opt."""
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        tour = points[hull.vertices]
        return self._2opt(tour)

    def _2opt(self, tour):
        """Optimización local 2-opt."""
        improved = True
        while improved:
            improved = False
            for i in range(len(tour)-1):
                for j in range(i+2, len(tour)):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%len(tour)]
                    if (np.linalg.norm(a-c) + np.linalg.norm(b-d) < 
                        np.linalg.norm(a-b) + np.linalg.norm(c-d)):
                        tour[i+1:j+1] = tour[j:i:-1]
                        improved = True
        return tour

    def _3opt(self, tour):
        """Optimización 3-opt (versión simplificada)."""
        best_tour = tour
        best_dist = self._calculate_distance(tour)
        
        for _ in range(self.params['classic']['3opt_iters']):
            for i in range(len(tour)):
                for j in range(i+2, len(tour)):
                    for k in range(j+2, len(tour)):
                        new_tour = np.concatenate([
                            tour[:i+1],
                            tour[j:k+1][::-1],
                            tour[i+1:j][::-1],
                            tour[k+1:]
                        ])
                        new_dist = self._calculate_distance(new_tour)
                        if new_dist < best_dist:
                            best_tour, best_dist = new_tour, new_dist
        
        return best_tour

    def _brute_force(self, points):
        """Fuerza bruta para clusters pequeños (N ≤ 8)."""
        from itertools import permutations
        min_dist = float('inf')
        best_tour = points
        for perm in permutations(points):
            dist = sum(np.linalg.norm(perm[i]-perm[(i+1)%len(perm)]) 
                     for i in range(len(perm)))
            if dist < min_dist:
                min_dist = dist
                best_tour = perm
        return np.array(best_tour)

    def _assemble_global_tour(self, clusters, sub_tours):
        """Combina subtours con enfoque greedy."""
        full_tour = np.concatenate(sub_tours)
        return self._2opt(full_tour)

    def _classic_optimization(self, tour, time_limit):
        """Refinamiento final con 3-opt."""
        start_time = time.time()
        best_tour = tour
        best_dist = self._calculate_distance(tour)
        
        while time.time() - start_time < time_limit:
            new_tour = self._3opt(best_tour)
            new_dist = self._calculate_distance(new_tour)
            if new_dist < best_dist:
                best_tour, best_dist = new_tour, new_dist
            else:
                break
        
        return best_tour

    def _calculate_distance(self, tour):
        """Calcula la distancia total de la ruta."""
        return sum(np.linalg.norm(tour[i]-tour[(i+1)%len(tour)]) 
                 for i in range(len(tour)))

    def plot_tour(self, tour):
        """Visualiza la ruta óptima."""
        plt.figure(figsize=(10,6))
        plt.plot(tour[:,0], tour[:,1], 'bo-')
        for i, (x, y) in enumerate(tour):
            plt.text(x, y, str(i), color='red', fontsize=12)
        plt.title(f"Ruta Óptima (Distancia: {self._calculate_distance(tour):.2f})")
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (10 ciudades aleatorias)
    np.random.seed(42)
    cities = np.random.rand(10, 2) * 100
    
    solver = TSPSolver(cities)
    tour, distance = solver.solve(time_limit=120)  # 2 minutos
    
    print(f"Distancia total: {distance:.2f}")
    solver.plot_tour(tour)
