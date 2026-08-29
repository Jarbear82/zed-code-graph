# fCoSE Layout Architectural Proposal

This document outlines the proposed architecture for porting the Cytoscape fCoSE algorithm to Rust for the Zed Graph Lens.

## 1. Data-Oriented Graph Structures

To avoid `Rc<RefCell<...>>` and satisfy the borrow checker, we will use a **Parallel Data Structure** approach. The layout engine will operate on a `LayoutGraph` which mirrors the necessary state from `GraphEngine` into flat, cache-friendly vectors.

```rust
/// Internal representation of the graph for layout calculation.
/// Optimized for cache locality and ease of iteration.
pub struct LayoutGraph {
    // Structural Data
    pub nodes: Vec<LayoutNode>,
    pub edges: Vec<LayoutEdge>,
    
    // Physics State (Flat Vectors for speed)
    pub positions: Vec<Point<f32>>,
    pub velocities: Vec<Point<f32>>,
    pub forces: Vec<Point<f32>>,
    pub masses: Vec<f32>,
    pub sizes: Vec<Size<f32>>,
    
    // Hierarchy (Index-based)
    pub parents: Vec<Option<usize>>,
    pub children: Vec<Vec<usize>>,
}

pub struct LayoutNode {
    pub id: NodeId, // Original ID from GraphEngine
    pub is_fixed: bool,
}

pub struct LayoutEdge {
    pub source: usize, // Internal index
    pub target: usize, // Internal index
    pub ideal_length: f32,
}
```

## 2. Proposed Module Tree

The code will be organized into logical units within `crates/graph_lens/src/layouts/`:

```
src/layouts/
├── mod.rs          # Layout trait and shared types
├── fcose.rs        # Main fCoSE coordinator (orchestrates spectral + force)
├── math.rs         # Linear algebra: Vector2D, Matrix operations, Eigen solvers
├── spectral.rs     # Spectral Initialization (Random-walk Laplacian)
├── force.rs        # Physics Engine: Spring, Repulsion, Gravity calculations
├── quadtree.rs     # Optimized Barnes-Hut implementation for Repulsion
└── state.rs        # Simulation parameters (Temperature, Cooling, Constraints)
```

## 3. Separation of Phases

The layout process is split into two distinct execution models:

### Phase A: Spectral Initialization (Synchronous Pre-pass)
1. **Laplacian Matrix Construction**: Build the Random-Walk Laplacian from the graph topology.
2. **Eigenvector Calculation**: Use Power Iteration + Gram-Schmidt to find the 2nd and 3rd eigenvectors.
3. **Dispersal**: Assign initial `(x, y)` based on these vectors to minimize edge crossings globally.
4. **Component Packing**: If the graph is disconnected, use a "Strip Packing" algorithm to arrange sub-components.

### Phase B: Force-Directed Relaxation (Asynchronous Loop)
1. **Input**: Initial positions from Phase A.
2. **Loop**: Runs in a GPUI background task.
   - **Calculate Forces**: Parallel iteration over edges (springs) and QuadTree (repulsion).
   - **Apply Constraints**: Resolve compound node boundaries and fixed nodes.
   - **Integrate**: Update `velocity` and `position` using a cooling factor (Simulated Annealing).
   - **Sync**: Update the `GraphEngine` with new positions every N ticks.

## 4. GPUI Integration & Physics Update

The physics simulation will be decoupled from the UI thread using GPUI's concurrency primitives:

1. **State Ownership**: `GraphEngine` remains the source of truth for the UI.
2. **The Physics Task**:
   ```rust
   cx.spawn(|this, mut cx| async move {
       let mut layout = fCoSELayout::prepare(&graph_engine);
       loop {
           layout.tick(); // Perform 1 or more simulation steps
           
           // Apply updates back to GraphEngine
           this.update(&mut cx, |this, cx| {
               layout.sync_to(&mut this.state.graph_engine);
               cx.notify();
           })?;
           
           // Throttle to prevent CPU starvation
           cx.background_executor().timer(Duration::from_millis(16)).await;
       }
   })
   ```

## 5. Core Traits

```rust
pub trait Layout {
    /// Initialize the layout with the current graph state.
    fn initialize(&mut self, engine: &GraphEngine);
    
    /// Perform a single iteration of the physics engine.
    fn tick(&mut self);
    
    /// Sync internal positions back to the GraphEngine.
    fn sync_to(&self, engine: &mut GraphEngine);
    
    /// Returns true if the simulation has reached equilibrium (temperature < threshold).
    fn is_stable(&self) -> bool;
}
```
