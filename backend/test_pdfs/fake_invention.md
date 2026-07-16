# Temporal-Spatial Resonance Fields for Autonomous Micro-Drone Swarm Coordination

**Authors:** J. Kirkpatrick, M. Navarro, S. Okonkwo  
**Date:** April 2026  
**Affiliation:** Department of Aerospace Engineering, Fictional University

## Abstract

We present Temporal-Spatial Resonance Fields (TSRF), a novel method for coordinating autonomous micro-drone swarms without centralized communication. Unlike existing approaches that rely on broadcast consensus protocols (e.g., RAFT-based swarm consensus) or pre-computed trajectory libraries, TSRF enables each drone to independently compute a shared coordination field by combining three techniques we introduce:

1. **Resonance Embedding**: Each drone maintains a 64-dimensional embedding vector that encodes its current state (position, velocity, battery, payload) and broadcasts it via ultra-wideband (UWB) pulses at 10Hz. Unlike standard state-sharing, the embedding is trained via a self-supervised contrastive loss that makes nearby drones' embeddings converge and distant drones' embeddings diverge — creating an implicit spatial topology without explicit position exchange.

2. **Temporal Phase Alignment**: Rather than synchronizing clocks (which requires GPS or NTP infrastructure), each drone derives its local phase from the interference pattern of received UWB pulses. We show that with >=4 neighbors, the phase converges to a globally consistent temporal reference within 200ms, enabling coordinated maneuvers without any external timing source.

3. **Field Gradient Navigation**: Each drone computes a local approximation of the global resonance field using only its own embedding and those of its 1-hop neighbors. We prove that the gradient of this field points toward the optimal formation position, enabling decentralized formation control with provable convergence guarantees under our assumptions (bounded velocity, connected graph, Lipschitz-continuous field).

## Method

### 2.1 Resonance Embedding Training

We train the embedding network (a 3-layer MLP with ReLU activations, input: 12-dim state vector, output: 64-dim embedding) using a modified NT-Xent contrastive loss. Positive pairs are drones within 5m of each other; negative pairs are drones >20m apart. The key innovation is that training happens **online during flight** — each drone updates its embedding network every 100ms using the embeddings received from neighbors in the last second. This means the swarm's coordination improves over time without any pre-training phase.

The loss function is:

L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

where sim is cosine similarity, τ=0.07 is the temperature, z_i is the drone's own embedding, z_j is a positive pair (neighbor within 5m), and z_k iterates over all received embeddings.

### 2.2 Temporal Phase Alignment

Each drone emits UWB pulses at its local 10Hz rate. When it receives pulses from neighbors, it computes the phase difference between its own emission cycle and the received pulses. We define the phase error as:

φ_err = Σ_j w_j * sin(θ_i - θ_j)

where θ_i is drone i's phase, θ_j is the estimated phase of neighbor j (from pulse arrival times), and w_j = 1/d_ij (inverse distance weighting, d_ij from UWB ranging). Each drone adjusts its phase by Δθ = -η * φ_err with η=0.3.

We prove (Theorem 1) that this Kuramoto-type coupling converges to global phase synchronization for any connected graph topology, with convergence time O(1/λ_2) where λ_2 is the algebraic connectivity of the neighbor graph.

### 2.3 Field Gradient Navigation

Given synchronized phases and converged embeddings, each drone computes the resonance field at its position:

F(x_i) = Σ_j K(z_i, z_j) * G(x_i - x_j)

where K is a learned kernel on the embedding space (parameterized as a 2-layer MLP) and G is a Gaussian spatial kernel. The gradient ∇F(x_i) gives the direction toward the optimal formation position.

We prove (Theorem 2) that under our assumptions, following -∇F converges to the Voronoi-optimal formation in O(n log n) steps.

## Experiments

We validate TSRF on a swarm of 32 Crazyflie 2.1 micro-drones in a 10m×10m×3m indoor arena with OptiTrack for ground truth (not used by the drones). Results:

- **Formation convergence**: 32 drones achieve target hexagonal formation in 4.2s (σ=0.8s) from random initial positions, compared to 12.3s for RAFT-consensus and 8.1s for potential field methods.
- **Robustness**: Removing 8 drones mid-flight (simulating failures) causes the remaining 24 to re-converge to optimal formation in 2.1s.
- **Online learning**: Embedding quality (measured by neighbor classification accuracy) improves from 72% at t=0 to 96% at t=30s of flight.
- **No GPS/NTP**: Phase alignment achieves <1ms global synchronization error using only UWB, compared to 5-10ms for GPS-based synchronization.

## Conclusion

TSRF represents the first fully decentralized, infrastructure-free coordination method for micro-drone swarms that simultaneously solves spatial coordination, temporal synchronization, and formation control using a unified resonance field framework. The key novelty is the combination of online contrastive embedding learning with Kuramoto-type phase coupling, enabling emergent coordination without any pre-training, centralized communication, or external infrastructure.
