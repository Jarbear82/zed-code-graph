use crate::graph_engine::{GraphEngine, NodeId};
use crate::layouts::math::{self, Matrix};
use std::collections::HashMap;

pub struct SpectralLayout {
    pub iterations: usize,
}

impl SpectralLayout {
    pub fn new() -> Self {
        Self { iterations: 50 }
    }

    /// Computes initial positions for nodes based on the Random-Walk Graph Laplacian,
    /// handling disconnected components and tiling isolated nodes.
    pub fn apply(&self, engine: &mut GraphEngine) {
        let components = engine.detect_components();
        if components.is_empty() {
            return;
        }

        let mut singletons = Vec::new();
        let mut clusters = Vec::new();

        for comp in components {
            if comp.len() == 1 {
                singletons.push(comp[0]);
            } else {
                clusters.push(comp);
            }
        }

        let mut component_bounds = Vec::new();

        // 1. Layout each cluster independently
        for cluster in &clusters {
            let bounds = self.layout_cluster(engine, cluster);
            component_bounds.push(bounds);
        }

        // 2. Tile singletons
        if !singletons.is_empty() {
            let bounds = self.tile_singletons(engine, &singletons);
            component_bounds.push(bounds);
        }

        // 3. Arrange component centers (Strip Packing)
        self.pack_components(engine, &clusters, &singletons, &component_bounds);
    }

    fn layout_cluster(
        &self,
        engine: &mut GraphEngine,
        cluster: &[NodeId],
    ) -> (gpui::Point<f32>, gpui::Size<f32>) {
        let n = cluster.len();
        let mut index_map: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        for (i, &id) in cluster.iter().enumerate() {
            index_map.insert(id, i);
        }

        let mut adj = vec![Vec::new(); n];
        for edge in &engine.edges {
            if let (Some(&u), Some(&v)) = (index_map.get(&edge.source), index_map.get(&edge.target))
            {
                adj[u].push(v);
                adj[v].push(u);
            }
        }

        // Include parent-child links
        for (i, &id) in cluster.iter().enumerate() {
            let node = &engine.nodes[id.0];
            if let Some(parent) = node.parent
                && let Some(&p) = index_map.get(&parent)
                && !adj[i].contains(&p)
            {
                adj[i].push(p);
                adj[p].push(i);
            }
        }

        let mut m = Matrix::new(n, n);
        for (i, neighbors) in adj.iter().enumerate() {
            let degree = neighbors.len() as f32;
            if degree > 0.0 {
                for &j in neighbors {
                    m.set(i, j, 1.0 / degree);
                }
            } else {
                m.set(i, i, 1.0);
            }
        }

        let mut v0 = vec![1.0; n];
        math::normalize(&mut v0);

        let mut v1 = math::random_unit_vector(n);
        for _ in 0..self.iterations {
            v1 = m.multiply_vec(&v1);
            math::orthogonalize_unit(&mut v1, &v0);
            math::normalize(&mut v1);
        }

        let mut v2 = math::random_unit_vector(n);
        for _ in 0..self.iterations {
            v2 = m.multiply_vec(&v2);
            math::orthogonalize_unit(&mut v2, &v0);
            math::orthogonalize_unit(&mut v2, &v1);
            math::normalize(&mut v2);
        }

        let optimal_spacing = 150.0;
        let scale = (n as f32).sqrt() * optimal_spacing;

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for (i, &id) in cluster.iter().enumerate() {
            let node = &mut engine.nodes[id.0];
            if !node.is_fixed {
                // Assign local positions centered at (0,0)
                node.position.x = v1[i] * scale;
                node.position.y = v2[i] * scale;

                min_x = min_x.min(node.position.x);
                max_x = max_x.max(node.position.x + node.size.width);
                min_y = min_y.min(node.position.y);
                max_y = max_y.max(node.position.y + node.size.height);
            }
        }

        (
            gpui::Point {
                x: (min_x + max_x) / 2.0,
                y: (min_y + max_y) / 2.0,
            },
            gpui::Size {
                width: (max_x - min_x).max(100.0),
                height: (max_y - min_y).max(100.0),
            },
        )
    }

    fn tile_singletons(
        &self,
        engine: &mut GraphEngine,
        singletons: &[NodeId],
    ) -> (gpui::Point<f32>, gpui::Size<f32>) {
        let n = singletons.len();
        if n == 0 {
            return (gpui::Point::default(), gpui::Size::default());
        }

        // Favor a landscape aspect ratio (roughly 1.6:1)
        let ratio = 1.6;
        let cols = (n as f32 * ratio).sqrt().ceil().max(1.0) as usize;
        let spacing_x = 220.0;
        let spacing_y = 120.0;

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for (i, &id) in singletons.iter().enumerate() {
            let node = &mut engine.nodes[id.0];
            if !node.is_fixed {
                let row = i / cols;
                let col = i % cols;
                node.position.x = col as f32 * spacing_x;
                node.position.y = row as f32 * spacing_y;

                min_x = min_x.min(node.position.x);
                max_x = max_x.max(node.position.x + node.size.width);
                min_y = min_y.min(node.position.y);
                max_y = max_y.max(node.position.y + node.size.height);
            }
        }

        (
            gpui::Point {
                x: (min_x + max_x) / 2.0,
                y: (min_y + max_y) / 2.0,
            },
            gpui::Size {
                width: (max_x - min_x).max(100.0),
                height: (max_y - min_y).max(100.0),
            },
        )
    }

    fn pack_components(
        &self,
        engine: &mut GraphEngine,
        clusters: &[Vec<NodeId>],
        singletons: &[NodeId],
        bounds: &[(gpui::Point<f32>, gpui::Size<f32>)],
    ) {
        let padding = 300.0;
        let mut current_x = 0.0f32;
        let mut current_y = 0.0f32;
        let mut max_h_in_row = 0.0f32;

        // Calculate a better wrap width based on the total area and desired aspect ratio (1.6:1)
        let total_area: f32 = bounds
            .iter()
            .map(|(_, size)| (size.width + padding) * (size.height + padding))
            .sum();
        let target_ratio = 1.6;
        let max_row_w = (total_area * target_ratio).sqrt().max(1200.0);

        for (i, (center, size)) in bounds.iter().enumerate() {
            if current_x + size.width > max_row_w && current_x > 0.0 {
                current_x = 0.0;
                current_y += max_h_in_row + padding;
                max_h_in_row = 0.0;
            }

            // Offset the nodes in this component to their packed position
            let offset_x = current_x - (center.x - size.width / 2.0);
            let offset_y = current_y - (center.y - size.height / 2.0);

            if i < clusters.len() {
                for &id in &clusters[i] {
                    let node = &mut engine.nodes[id.0];
                    if !node.is_fixed {
                        node.position.x += offset_x;
                        node.position.y += offset_y;
                        node.velocity = gpui::Point::default();
                    }
                }
            } else {
                // Packing singletons (they are the last "component" in bounds)
                for &id in singletons {
                    let node = &mut engine.nodes[id.0];
                    if !node.is_fixed {
                        node.position.x += offset_x;
                        node.position.y += offset_y;
                        node.velocity = gpui::Point::default();
                    }
                }
            }

            current_x += size.width + padding;
            max_h_in_row = max_h_in_row.max(size.height);
        }

        // Center the entire graph on the canvas
        let mut total_min_x = f32::MAX;
        let mut total_max_x = f32::MIN;
        let mut total_min_y = f32::MAX;
        let mut total_max_y = f32::MIN;

        for id in &engine.active_nodes {
            let node = &engine.nodes[id.0];
            total_min_x = total_min_x.min(node.position.x);
            total_max_x = total_max_x.max(node.position.x + node.size.width);
            total_min_y = total_min_y.min(node.position.y);
            total_max_y = total_max_y.max(node.position.y + node.size.height);
        }

        let graph_center_x = (total_min_x + total_max_x) / 2.0;
        let graph_center_y = (total_min_y + total_max_y) / 2.0;
        let canvas_center_x = engine.bounds.x / 2.0;
        let canvas_center_y = engine.bounds.y / 2.0;

        let final_offset_x = canvas_center_x - graph_center_x;
        let final_offset_y = canvas_center_y - graph_center_y;

        for id in &engine.active_nodes {
            let node = &mut engine.nodes[id.0];
            if !node.is_fixed {
                node.position.x += final_offset_x;
                node.position.y += final_offset_y;
            }
        }
    }
}
