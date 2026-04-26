# Decentralized Adaptive Coverage Control
### ROS2 Implementation on TurtleBot3

Implementation of the paper **"Decentralized, Adaptive Coverage Control for Networked Robots"** by Schwager, Rus & Slotine — MIT (2007).

---

## What This Does

A group of TurtleBot3 robots deployed in an unknown environment autonomously spread out and concentrate near regions of high sensory importance — with no central controller and no prior knowledge of the environment.

Each robot:
- Learns an approximation of the sensory function from its own sensor readings
- Shares its learned parameters with neighbouring robots every 0.1 seconds
- Moves toward the centroid of its Voronoi region
- Collectively converges to a near-optimal sensing formation

---

## The Core Idea

The environment has a sensory function φ(q) — in this simulation, a Gaussian representing a toxic gas concentration peaked at position (1, 1):

```
φ(x, y) = exp( −((x − 1)² + (y − 1)²) )
```

No robot is given this function. Each robot only measures φ at its own position once every 0.1 seconds. Through the adaptive law and consensus mechanism, the robots collectively learn where the important regions are and position themselves optimally.

---

## Project Structure

```
adaptive_coverage2/
│
├── adaptive_coverage2/
│   ├── geometry/
│   │   ├── basis_function.py       # K(q) — polynomial basis functions
│   │   ├── compute_density.py      # φ̂ᵢ(q) = K(q)ᵀ âᵢ
│   │   ├── compute_centroid.py     # Voronoi centroid via local grid integration
│   │   ├── compute_laplacian.py    # Consensus correction term
│   │   ├── adaptive_law.py         # Λᵢ, λᵢ accumulation + projection
│   │   └── projection.py           # Parameter projection operator
│   │
│   ├── coverage_controller.py      # Main control loop — runs at 10 Hz
│   └── neighbor_exchange.py        # ROS2 communication node
│
├── launch/
│   └── coverage_launch.py
├── package.xml
└── setup.py
```

---

## How It Works

### 1. Sensory Function Approximation

Each robot approximates the unknown φ(q) as:

```
φ̂ᵢ(q, t) = K(q)ᵀ âᵢ(t)
```

where K(q) is a fixed polynomial basis:

```python
K(x, y) = [1, x, y, x², y², xy, x³, y³, x²y]   # m = 9
```

and âᵢ(t) is a 9-dimensional weight vector learned from sensor data.

### 2. Voronoi Partition

The workspace is divided among robots automatically. A point q belongs to robot i if robot i is the closest robot to q:

```
Vᵢ = { q : ‖q − pᵢ‖ ≤ ‖q − pⱼ‖  for all j ≠ i }
```

Voronoi regions reshape continuously as robots move.

### 3. Control Law

Each robot moves toward the centroid of its estimated Voronoi region:

```
ṗᵢ = K × (Ĉᵥᵢ − pᵢ)
```

In code:
```python
u = -self.kp * (self.position - c_i)
```

### 4. Adaptation Law

Weights update every 0.1 seconds using Euler integration:

```
â_new = â_old + â_dot × Δt
```

where the rate of change is:

```
â_dot = −γ(Λᵢ âᵢ − λᵢ) + consensus correction
```

Two quantities accumulate the robot's sensing history:

```python
# Λᵢ — excitation matrix (memory of where robot has been)
self.Lambda += phi @ phi.T * dt

# λᵢ — information vector (memory of what sensor read there)
self.lambda_vec += phi * sensor_reading * dt
```

### 5. Consensus

Robots share their â vectors with neighbours every 0.1 seconds. Each robot is pulled toward its neighbours' estimates:

```python
a_dot -= zeta * w * (a_hat - a_j)   # for each neighbour j
```

This allows knowledge from one robot's trajectory to propagate to all robots in the network.

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `kp` | 1.0 | Proportional control gain |
| `gamma` | 0.5 | Adaptation learning rate |
| `zeta` | 0.1 | Consensus gain |
| `dt` | 0.1 | Control loop period (seconds) |
| `n_basis` | 9 | Number of basis functions |
| `a_min` | 0.01 | Minimum parameter value (projection floor) |
| `grid` | 0.6 m | Local Voronoi integration window radius |
| `res` | 0.05 m | Grid resolution for centroid computation |

---

## Differences From The Paper

| Aspect | Paper | This Implementation |
|---|---|---|
| Basis functions | Gaussians — centers and widths chosen by engineer | Polynomial monomials — no placement decisions |
| Voronoi integration | Full Vᵢ — global centroid | Local 0.6m × 0.6m window — real-time constraint |
| Edge weights lᵢⱼ | Shared Voronoi edge length | lᵢⱼ = 1 for all neighbours (Remark 5) |
| Fᵢ term in â_dot | Included — compensates centroid error | Omitted — too expensive for 10 Hz loop |
| Projection | Smart gate on derivative â_dot | Hard clamp on â after update |
| Robot dynamics | Integrator ṗᵢ = uᵢ | Unicycle — rotation matrix to (v, ω) |

---

## What The Paper Proves

The paper provides a Lyapunov-based proof that the system is guaranteed to converge. The Lyapunov function:

```
V = H + Σᵢ ½ ãᵢᵀ Γ⁻¹ ãᵢ
```

combines the sensing cost H and the parameter errors ãᵢ = âᵢ − a into one scalar that can only decrease over time. By Barbalat's Lemma, V̇ → 0, which forces:

- pᵢ → Ĉᵥᵢ — robots reach their estimated centroids
- φ̂ᵢ → φ over visited regions — robots learn the true sensory function
- âᵢ → âⱼ for all i, j — all robots agree on the same parameters

**Note:** The Lyapunov proof is a mathematical guarantee — it is not implemented in the code. The code implements the control law and adaptation law. The proof guarantees those laws produce convergence.

---

## Limitations

- Each robot accurately knows φ only over regions it physically visited
- Through consensus it gains partial knowledge of regions visited by neighbours
- Regions never visited by any robot remain unknown to the entire network
- The formation is near-optimal — true optimality requires sufficient richness of collective trajectories
- The true_density function simulates the sensor — replace with a real sensor callback for hardware deployment

---

## Requirements

```
ROS2 Humble or later
TurtleBot3
Python 3.8+
numpy
tf_transformations
```

---

## Installation

```bash
cd ~/ros2_ws/src
git clone <your-repo-url>
cd ~/ros2_ws
colcon build --packages-select adaptive_coverage2
source install/setup.bash
```

---

## Running

```bash
# Launch the coverage controller for two robots
ros2 launch adaptive_coverage2 coverage_launch.py
```

---

## Reference

Schwager, M., Rus, D., & Slotine, J. J. (2007).  
**Decentralized, Adaptive Coverage Control for Networked Robots.**  
MIT Computer Science and Artificial Intelligence Laboratory.  
*Revised July 2008.*

