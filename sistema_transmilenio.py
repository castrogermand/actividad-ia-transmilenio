import heapq
import math

# ==========================================
# 1. BASE DE CONOCIMIENTO (DATOS DEL SISTEMA)
# ==========================================

# Coordenadas aproximadas (x, y) para calcular distancias heurísticas
# Representación simplificada del eje Norte-Centro (Calle 80 / Av. Caracas)
estaciones = {
    "Terminal": (0, 100),
    "Toberin": (5, 95),
    "Prado": (10, 90),
    "Mazuren": (15, 85),
    "Alcalá": (20, 80),
    "Suba_A": (25, 75), # Suba - Calle 75
    "Suba_C": (30, 70), # Suba - Calle 70
    "Calle_80": (35, 65),
    "Gran_Estacion": (40, 60),
    "Salitre": (45, 55),
    "Av_Boyaca": (50, 50),
    "Flores": (55, 45),
    "Polo": (60, 40),
    "San_Luis": (65, 35),
    "Museo": (70, 30),
    "Las_Aguas": (75, 25),
    "Universidades": (80, 20),
    "Avenida_Jimenez": (85, 15)
}

# Conexiones directas y sus costos (distancia en km aprox)
# Grafo no dirigido (A -> B es igual a B -> A)
conexiones = {
    "Terminal": {"Toberin": 2.5},
    "Toberin": {"Terminal": 2.5, "Prado": 2.0},
    "Prado": {"Toberin": 2.0, "Mazuren": 2.0},
    "Mazuren": {"Prado": 2.0, "Alcalá": 2.0},
    "Alcalá": {"Mazuren": 2.0, "Suba_A": 2.5},
    "Suba_A": {"Alcalá": 2.5, "Suba_C": 2.0},
    "Suba_C": {"Suba_A": 2.0, "Calle_80": 2.0},
    "Calle_80": {"Suba_C": 2.0, "Gran_Estacion": 2.0},
    "Gran_Estacion": {"Calle_80": 2.0, "Salitre": 2.0},
    "Salitre": {"Gran_Estacion": 2.0, "Av_Boyaca": 2.5},
    "Av_Boyaca": {"Salitre": 2.5, "Flores": 2.0},
    "Flores": {"Av_Boyaca": 2.0, "Polo": 2.0},
    "Polo": {"Flores": 2.0, "San_Luis": 2.0},
    "San_Luis": {"Polo": 2.0, "Museo": 1.5},
    "Museo": {"San_Luis": 1.5, "Las_Aguas": 1.5},
    "Las_Aguas": {"Museo": 1.5, "Universidades": 1.5},
    "Universidades": {"Las_Aguas": 1.5, "Avenida_Jimenez": 1.5},
    "Avenida_Jimenez": {"Universidades": 1.5}
}

# ==========================================
# 2. SISTEMA BASADO EN REGLAS (LÓGICA)
# ==========================================

class MotorReglas:
    """Sistema experto que valida si un movimiento es permitido"""
    
    def validar_movimiento(self, origen, destino, hora_pico=False):
        """
        Aplica reglas lógicas para determinar si se puede ir de A a B
        Regla 1: Debe existir conexión física
        Regla 2: Restricción de hora pico (simulada)
        Regla 3: No retroceder en la ruta óptima (heurística simple)
        """
        razones = []
        permitido = True

        # Regla 1: Existencia de conexión
        if origen not in conexiones or destino not in conexiones[origen]:
            return False, ["Regla 1 Violada: No hay ruta directa entre estas estaciones."]

        # Regla 2: Restricción operativa (Ejemplo: Estación cerrada por mantenimiento)
        # Simulamos que 'Suba_C' tiene restricciones en hora pico
        if hora_pico and destino == "Suba_C":
            return False, ["Regla 2 Violada: Estación con restricción en hora pico."]

        # Regla 3: Validación de dirección (hacia el destino final)
        # En un sistema real esto sería más complejo, aquí solo validamos conectividad
        
        if permitido:
            razones.append("Movimiento válido según base de conocimientos.")
            
        return permitido, razones

# ==========================================
# 3. ALGORITMO DE BÚSQUEDA A* (HEURÍSTICA)
# ==========================================

def heuristic(a, b):
    """Calcula la distancia euclidiana como heurística (h(n))"""
    x1, y1 = estaciones[a]
    x2, y2 = estaciones[b]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def busqueda_a_estrella(inicio, fin, motor_reglas):
    """
    Implementación del algoritmo A*
    Retorna: (camino, costo_total, nodos_visitados)
    """
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0, inicio))
    
    came_from = {} # Para reconstruir el camino
    g_score = {estacion: float('inf') for estacion in estaciones}
    g_score[inicio] = 0
    
    f_score = {estacion: float('inf') for estacion in estaciones}
    f_score[inicio] = heuristic(inicio, fin)
    
    visitados = set()
    
    while cola_prioridad:
        current_f, current = heapq.heappop(cola_prioridad)
        
        if current == fin:
            # Reconstruir camino
            camino = []
            while current in came_from:
                camino.append(current)
                current = came_from[current]
            camino.append(inicio)
            camino.reverse()
            return camino, g_score[fin], len(visitados)
        
        visitados.add(current)
        
        if current not in conexiones:
            continue
            
        for vecino, costo in conexiones[current].items():
            # Consultar al sistema de reglas antes de moverse
            permitido, _ = motor_reglas.validar_movimiento(current, vecino)
            
            if not permitido:
                continue # Saltar este vecino si las reglas lo prohíben

            tentative_g = g_score[current] + costo
            
            if tentative_g < g_score[vecino]:
                came_from[vecino] = current
                g_score[vecino] = tentative_g
                f_score[vecino] = tentative_g + heuristic(vecino, fin)
                
                if vecino not in [i[1] for i in cola_prioridad]:
                    heapq.heappush(cola_prioridad, (f_score[vecino], vecino))
    
    return None, float('inf'), len(visitados) # No se encontró ruta

# ==========================================
# 4. INTERFAZ DE USUARIO Y EJECUCIÓN
# ==========================================

def mostrar_ruta(camino, costo):
    print("\n" + "="*40)
    print("🚌 RUTA ÓPTIMA ENCONTRADA (ALGORITMO A*)")
    print("="*40)
    
    if not camino:
        print("❌ No se encontró una ruta válida bajo las reglas actuales.")
        return

    print(f"📍 Origen: {camino[0]}")
    print(f"🏁 Destino: {camino[-1]}")
    print(f"🛣️  Recorrido: {' -> '.join(camino)}")
    print(f"📏 Distancia Total: {costo:.2f} km")
    print(f"⏱️  Tiempo Estimado: {costo * 2.5:.0f} minutos") # Approx 2.5 min/km
    print(f"🎫 Costo Pasaje: $2.950 COP")
    print("="*40)

def main():
    print("🤖 SISTEMA INTELIGENTE DE TRANSPORTE - TRANSMILENIO")
    print("Basado en Reglas Lógicas y Búsqueda Heurística A*")
    print("-" * 50)
    
    # Inicializar motor de reglas
    motor = MotorReglas()
    
    # Mostrar estaciones disponibles
    print("Estaciones disponibles:")
    lista_estaciones = list(estaciones.keys())
    for i, est in enumerate(lista_estaciones):
        print(f"{i+1}. {est}")
    
    try:
        origen_nombre = input("\nIngrese el nombre de la estación de ORIGEN: ").strip()
        destino_nombre = input("Ingrese el nombre de la estación de DESTINO: ").strip()
        
        # Validar que las estaciones existen
        if origen_nombre not in estaciones or destino_nombre not in estaciones:
            print("❌ Error: Una o ambas estaciones no existen en la base de datos.")
            return

        # Ejecutar búsqueda
        print("\n🔍 Calculando mejor ruta...")
        camino, costo, visitados = busqueda_a_estrella(origen_nombre, destino_nombre, motor)
        
        mostrar_ruta(camino, costo)
        
        print(f"\nℹ️  Nodos analizados durante la búsqueda: {visitados}")
        
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()