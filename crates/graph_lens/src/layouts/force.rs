use crate::constants::*;
use crate::graph_engine::{GraphEngine, LayoutMode, LayoutQuality, NodeId};
use crate::quadtree::QuadTree;
use gpui::{Bounds, Point, Size};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct CompoundForceLayout {
    pub k: f32,
    pub temp: f32,
    pub forces: Vec<Point<f32>>, // Pre-allocated force buffer
    pub pruned_leaves: HashMap<NodeId, NodeId>, // Leaf -> Parent/Neighbor
}

impl CompoundForceLayout {
    pub fn new(node_count: usize, width: f32, height: f32) -> Self {
        let area = width * height;
        let k = (area / (node_count as f32).max(1.0)).sqrt();
        Self {
            k,
            temp: 1.0,
            forces: vec![Point::default(); node_count],
            pruned_leaves: HashMap::new(),
        }
    }

    pub fn update(&mut self, engine: &mut GraphEngine) {
        let cooling_factor = match engine.quality {
            LayoutQuality::Draft => 0.9,
            LayoutQuality::Default => COOLING_FACTOR,
            LayoutQuality::Proof => 0.995,
        };

        let min_temp = match engine.quality {
            LayoutQuality::Draft => 0.1,
            LayoutQuality::Default => MIN_TEMPERATURE,
            LayoutQuality::Proof => 0.001,
        };

        if self.temp < min_temp {
            if !self.pruned_leaves.is_empty() {
                self.grow_leaves(engine);
            }
            return;
        }

        let components = engine.detect_components();
        if components.is_empty() {
            return;
        }

        let mut active_ids = Vec::new();
        let mut singleton_set = HashSet::new();
        let mut component_map = HashMap::new();

        for (_comp_idx, comp) in components.iter().enumerate() {
            for &id in comp {
                if self.pruned_leaves.contains_key(&id) {
                    continue;
                }
                active_ids.push(id);
                component_map.insert(id, _comp_idx);
            }
        }

        for (_comp_idx, comp) in components.iter().enumerate() {
            let active_in_comp: Vec<_> = comp
                .iter()
                .filter(|id| !self.pruned_leaves.contains_key(id))
                .collect();
            if active_in_comp.len() == 1 {
                singleton_set.insert(*active_in_comp[0]);
            }
        }

        if active_ids.is_empty() {
            // If all nodes are pruned (unlikely but possible), just grow them back
            self.grow_leaves(engine);
            return;
        }

        let active_set: HashSet<NodeId> = active_ids.iter().copied().collect();

        // Calculate centers of mass for each component
        let mut component_coms = vec![Point::<f32>::default(); components.len()];
        let mut component_masses = vec![0.0f32; components.len()];

        for &id in &active_ids {
            let node = &engine.nodes[id.0];
            let comp_idx = component_map[&id];
            let mass = (node.size.width * node.size.height).sqrt().max(1.0);

            component_coms[comp_idx].x += (node.position.x + node.size.width / 2.0) * mass;
            component_coms[comp_idx].y += (node.position.y + node.size.height / 2.0) * mass;
            component_masses[comp_idx] += mass;
        }
        for i in 0..components.len() {
            if component_masses[i] > 0.0 {
                component_coms[i].x /= component_masses[i];
                component_coms[i].y /= component_masses[i];
            }
        }

        // Ensure forces buffer is large enough and reset it to zero
        if self.forces.len() < engine.nodes.len() {
            self.forces.resize(engine.nodes.len(), Point::default());
        }
        for force in self.forces.iter_mut() {
            *force = Point::default();
        }

        // ==========================================
        // 1. Repulsion (Barnes-Hut via QuadTree)
        // ==========================================
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for &id in &active_ids {
            let node = &engine.nodes[id.0];
            min_x = min_x.min(node.position.x);
            max_x = max_x.max(node.position.x);
            min_y = min_y.min(node.position.y);
            max_y = max_y.max(node.position.y);
        }

        let world_size = (max_x - min_x).max(max_y - min_y).max(100.0);
        let mut qt = QuadTree::new(Bounds {
            origin: Point {
                x: min_x - OVERLAP_DISTANCE_THRESHOLD,
                y: min_y - OVERLAP_DISTANCE_THRESHOLD,
            },
            size: Size {
                width: world_size + 100.0,
                height: world_size + 100.0,
            },
        });

        for &id in &active_ids {
            let node = &engine.nodes[id.0];

            let raw_mass = (node.size.width * node.size.height).sqrt() * 0.1;
            let mass = (raw_mass.ln().max(1.0) * 5.0).max(1.0);

            let center_x = node.position.x + node.size.width / 2.0;
            let center_y = node.position.y + node.size.height / 2.0;
            qt.insert(
                id,
                Point {
                    x: center_x,
                    y: center_y,
                },
                mass,
            );
        }

        let repulsion_forces: Vec<(NodeId, Point<f32>)> = active_ids
            .par_iter()
            .map(|&id| {
                let node = &engine.nodes[id.0];
                let center_pos = Point {
                    x: node.position.x + node.size.width / 2.0,
                    y: node.position.y + node.size.height / 2.0,
                };

                let f = qt.calculate_repulsion_for_point(
                    id,
                    &engine.nodes,
                    center_pos,
                    0.5,
                    REPULSION_STRENGTH,
                );
                (id, f)
            })
            .collect();

        for (id, f) in repulsion_forces {
            self.forces[id.0].x += f.x;
            self.forces[id.0].y += f.y;
        }

        // ==========================================
        // 1.5 Overlap Correction
        // ==========================================
        for i in 0..active_ids.len() {
            for j in i + 1..active_ids.len() {
                let id1 = active_ids[i];
                let id2 = active_ids[j];
                let n1 = &engine.nodes[id1.0];
                let n2 = &engine.nodes[id2.0];

                if crate::quadtree::is_lineage(id1, id2, &engine.nodes) {
                    continue;
                }

                let c1 = Point {
                    x: n1.position.x + n1.size.width / 2.0,
                    y: n1.position.y + n1.size.height / 2.0,
                };
                let c2 = Point {
                    x: n2.position.x + n2.size.width / 2.0,
                    y: n2.position.y + n2.size.height / 2.0,
                };

                let mut dx = c1.x - c2.x;
                let mut dy = c1.y - c2.y;

                if dx.abs() < OVERLAP_DISTANCE_THRESHOLD && dy.abs() < OVERLAP_DISTANCE_THRESHOLD {
                    let seed = (id1.0 as f32 * 0.618 + id2.0 as f32 * 0.382).sin();
                    let angle = seed * std::f32::consts::PI * 2.0;
                    dx = angle.cos() * PUSH_DISTANCE;
                    dy = angle.sin() * PUSH_DISTANCE;
                }

                let overlap_x =
                    (n1.size.width + n2.size.width) / 2.0 + COMPOUND_PADDING_OTHER - dx.abs();
                let overlap_y =
                    (n1.size.height + n2.size.height) / 2.0 + COMPOUND_PADDING_OTHER - dy.abs();

                if overlap_x > 0.0 && overlap_y > 0.0 {
                    let dist = (dx * dx + dy * dy).sqrt().max(0.1);

                    let overlap_amount = overlap_x.max(overlap_y);
                    let capped_push = (overlap_amount * OVERLAP_CORRECTION_STIFFNESS).min(20.0);

                    let fx = (dx / dist) * capped_push;
                    let fy = (dy / dist) * capped_push;

                    self.forces[id1.0].x += fx;
                    self.forces[id1.0].y += fy;
                    self.forces[id2.0].x -= fx;
                    self.forces[id2.0].y -= fy;
                }
            }
        }

        // ==========================================
        // 2. Attractive Forces (Springs)
        // ==========================================
        for edge in &engine.edges {
            let proxy_source = resolve_proxy_node(edge.source, &engine.nodes, &active_set);
            let proxy_target = resolve_proxy_node(edge.target, &engine.nodes, &active_set);

            if proxy_source == proxy_target {
                continue;
            }

            if active_set.contains(&proxy_source) && active_set.contains(&proxy_target) {
                let n1 = &engine.nodes[proxy_source.0];
                let n2 = &engine.nodes[proxy_target.0];

                let cx1 = n1.position.x + n1.size.width / 2.0;
                let cy1 = n1.position.y + n1.size.height / 2.0;
                let cx2 = n2.position.x + n2.size.width / 2.0;
                let cy2 = n2.position.y + n2.size.height / 2.0;

                let dx = cx2 - cx1;
                let dy = cy2 - cy1;
                let dist = (dx * dx + dy * dy).sqrt().max(0.1);

                let radius1 = (n1.size.width + n1.size.height) / 4.0;
                let radius2 = (n2.size.width + n2.size.height) / 4.0;

                let base_length = IDEAL_EDGE_LENGTH.min(self.k * 0.75);

                // --- NEW: LCA Edge Stretching ---
                // Calculate the shared depth. Edges between nodes in different subtrees
                // get stretched proportional to their distance to a common ancestor.
                let lca_depth = calculate_lca_depth(proxy_source, proxy_target, &engine.nodes);
                let dist_to_lca = (n1.depth - lca_depth) + (n2.depth - lca_depth);

                // Stretch cross-directory edges to allow space for compound boxes
                let stretch_factor = 1.0 + (dist_to_lca as f32 * 0.2);
                let dynamic_ideal_length = (base_length * stretch_factor) + radius1 + radius2;

                let displacement = dist - dynamic_ideal_length;

                let mut stiffness = SPRING_STRENGTH;

                let is_promoted = proxy_source != edge.source || proxy_target != edge.target;
                if is_promoted {
                    stiffness *= 0.15;
                } else if n1.kind == crate::graph_engine::NodeKind::Directory
                    || n1.kind == crate::graph_engine::NodeKind::Worktree
                    || n2.kind == crate::graph_engine::NodeKind::Directory
                    || n2.kind == crate::graph_engine::NodeKind::Worktree
                {
                    stiffness *= 3.0;
                }

                let force = displacement * stiffness;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;

                self.forces[proxy_source.0].x += fx;
                self.forces[proxy_source.0].y += fy;
                self.forces[proxy_target.0].x -= fx;
                self.forces[proxy_target.0].y -= fy;
            }
        }

        // ==========================================
        // 3. Integrate & Apply Gravity
        // ==========================================
        let viewport_center = Point {
            x: engine.bounds.x / 2.0,
            y: engine.bounds.y / 2.0,
        };

        for &id in &active_ids {
            // Tiling Optimization: Isolated nodes don't participate in physics
            // once placed by SpectralLayout.
            if singleton_set.contains(&id) {
                continue;
            }

            let force = self.forces[id.0];

            let (is_fixed, gx, gy) = {
                let node = &engine.nodes[id.0];
                let node_center_x = node.position.x + node.size.width / 2.0;
                let node_center_y = node.position.y + node.size.height / 2.0;

                let mut calc_gx = 0.0;
                let mut calc_gy = 0.0;

                if let Some(parent_id) = node.parent {
                    let parent = &engine.nodes[parent_id.0];
                    let p_cx = parent.position.x + parent.size.width / 2.0;
                    let p_cy = parent.position.y + parent.size.height / 2.0;

                    // Intra-component tether (parent-child)
                    calc_gx = (p_cx - node_center_x) * 0.05;
                    calc_gy = (p_cy - node_center_y) * 0.05;
                } else {
                    // Component-level gravity:
                    // 1. Local gravity towards the component's center of mass
                    let comp_idx = component_map[&id];
                    let com: Point<f32> = component_coms[comp_idx];

                    calc_gx += (com.x - node_center_x) * (GRAVITY_STRENGTH * 2.0);
                    calc_gy += (com.y - node_center_y) * (GRAVITY_STRENGTH * 2.0);

                    // 2. Global tether towards viewport center
                    calc_gx += (viewport_center.x - node_center_x) * (GRAVITY_STRENGTH * 0.5);
                    calc_gy += (viewport_center.y - node_center_y) * (GRAVITY_STRENGTH * 0.5);
                }

                (node.is_fixed, calc_gx, calc_gy)
            };

            if is_fixed {
                continue;
            }

            let node = &mut engine.nodes[id.0];
            let velocity_damping = 0.8;

            let raw_mass = (node.size.width * node.size.height).sqrt() * 0.1;
            let safe_mass = raw_mass.max(1.0);

            node.velocity.x = (node.velocity.x + (force.x / safe_mass) + gx) * velocity_damping;
            node.velocity.y = (node.velocity.y + (force.y / safe_mass) + gy) * velocity_damping;

            node.velocity.x = node.velocity.x.clamp(-VELOCITY_CLAMP, VELOCITY_CLAMP);
            node.velocity.y = node.velocity.y.clamp(-VELOCITY_CLAMP, VELOCITY_CLAMP);

            node.position.x += node.velocity.x * self.temp * node.heat;
            node.position.y += node.velocity.y * self.temp * node.heat;

            node.heat *= 0.98; // Decay local heat
        }

        if engine.layout_mode == LayoutMode::Compound {
            // ==========================================
            // 3.5 Dynamic Parent Resizing
            // ==========================================
            let mut new_parent_sizes: HashMap<NodeId, Size<f32>> = HashMap::new();
            let parent_ids: Vec<NodeId> = active_ids
                .iter()
                .filter(|&&id| {
                    let kind = &engine.nodes[id.0].kind;
                    *kind == crate::graph_engine::NodeKind::Directory
                        || *kind == crate::graph_engine::NodeKind::Worktree
                })
                .copied()
                .collect();

            for pid in parent_ids {
                let mut min_x = f32::MAX;
                let mut max_x = f32::MIN;
                let mut min_y = f32::MAX;
                let mut max_y = f32::MIN;
                let mut has_children = false;

                for &child_id in &active_ids {
                    let node = &engine.nodes[child_id.0];
                    if node.parent == Some(pid) {
                        min_x = min_x.min(node.position.x);
                        max_x = max_x.max(node.position.x + node.size.width);
                        min_y = min_y.min(node.position.y);
                        max_y = max_y.max(node.position.y + node.size.height);
                        has_children = true;
                    }
                }

                if has_children {
                    let parent = &engine.nodes[pid.0];
                    let target_w =
                        ((max_x - min_x) + COMPOUND_PADDING_OTHER * 2.0).max(parent.min_width);
                    let target_h = (max_y - min_y) + COMPOUND_PADDING_TOP + COMPOUND_PADDING_OTHER;

                    let target_w = target_w.max(PARENT_MIN_WIDTH);
                    let target_h = target_h.max(PARENT_MIN_HEIGHT);

                    let new_w = parent.size.width
                        + (target_w - parent.size.width) * PARENT_SIZE_LERP_FACTOR;
                    let new_h = parent.size.height
                        + (target_h - parent.size.height) * PARENT_SIZE_LERP_FACTOR;

                    new_parent_sizes.insert(
                        pid,
                        Size {
                            width: new_w,
                            height: new_h,
                        },
                    );
                }
            }

            for (pid, new_size) in new_parent_sizes {
                engine.nodes[pid.0].size = new_size;
            }

            // ==========================================
            // 4. Hierarchical Constraints
            // ==========================================
            let mut parent_info = HashMap::new();
            for &id in &active_ids {
                let node = &engine.nodes[id.0];
                if node.parent.is_none()
                    || node.kind == crate::graph_engine::NodeKind::Directory
                    || node.kind == crate::graph_engine::NodeKind::Worktree
                {
                    parent_info.insert(id, (node.position, node.size));
                }
            }

            // --- NEW: Spatial Constraints ---
            for constraint in &engine.constraints {
                match constraint {
                    crate::graph_engine::Constraint::Alignment { nodes, horizontal } => {
                        if nodes.is_empty() {
                            continue;
                        }
                        let mut sum = 0.0;
                        let mut count = 0;
                        for &id in nodes {
                            if active_set.contains(&id) {
                                let node = &engine.nodes[id.0];
                                sum += if *horizontal {
                                    node.position.y + node.size.height / 2.0
                                } else {
                                    node.position.x + node.size.width / 2.0
                                };
                                count += 1;
                            }
                        }

                        if count > 0 {
                            let avg = sum / count as f32;
                            for &id in nodes {
                                if active_set.contains(&id) {
                                    let node = &mut engine.nodes[id.0];
                                    if !node.is_fixed {
                                        if *horizontal {
                                            node.position.y = avg - node.size.height / 2.0;
                                        } else {
                                            node.position.x = avg - node.size.width / 2.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    crate::graph_engine::Constraint::RelativePlacement {
                        anchor,
                        subject,
                        offset,
                    } => {
                        if active_set.contains(anchor) && active_set.contains(subject) {
                            let anchor_node = &engine.nodes[anchor.0];
                            let target_pos = Point {
                                x: anchor_node.position.x + offset.x,
                                y: anchor_node.position.y + offset.y,
                            };
                            let subject_node = &mut engine.nodes[subject.0];
                            if !subject_node.is_fixed {
                                subject_node.position.x +=
                                    (target_pos.x - subject_node.position.x) * 0.1;
                                subject_node.position.y +=
                                    (target_pos.y - subject_node.position.y) * 0.1;
                            }
                        }
                    }
                }
            }

            for &id in &active_ids {
                let pid = engine.nodes[id.0].parent;
                if let Some(pid) = pid
                    && let Some(&(p_pos, p_size)) = parent_info.get(&pid)
                {
                    let node = &mut engine.nodes[id.0];

                    let min_x = p_pos.x + COMPOUND_PADDING_OTHER;
                    let max_x = p_pos.x + p_size.width - node.size.width - COMPOUND_PADDING_OTHER;
                    let min_y = p_pos.y + COMPOUND_PADDING_TOP;
                    let max_y = p_pos.y + p_size.height - node.size.height - COMPOUND_PADDING_OTHER;

                    let max_x = max_x.max(min_x);
                    let max_y = max_y.max(min_y);

                    node.position.x = node.position.x.clamp(min_x, max_x);
                    node.position.y = node.position.y.clamp(min_y, max_y);
                }
            }
        }

        self.temp *= cooling_factor;
    }

    pub fn reset_temperature(&mut self, engine: &GraphEngine) {
        self.temp = 1.0;
        self.prune_leaves(engine);
    }

    pub fn prune_leaves(&mut self, engine: &GraphEngine) {
        self.pruned_leaves.clear();

        // Only prune in Default or Proof quality to keep Draft fast
        if engine.quality == LayoutQuality::Draft {
            return;
        }

        let mut degree = vec![0; engine.nodes.len()];
        for edge in &engine.edges {
            if edge.source.0 < degree.len() && edge.target.0 < degree.len() {
                degree[edge.source.0] += 1;
                degree[edge.target.0] += 1;
            }
        }

        for id in &engine.active_nodes {
            let node = &engine.nodes[id.0];
            // Only prune File/Outline nodes that have no edges (only connected to parent)
            if (node.kind == crate::graph_engine::NodeKind::File
                || node.kind == crate::graph_engine::NodeKind::Outline)
                && degree[id.0] == 0
            {
                if let Some(parent_id) = node.parent {
                    self.pruned_leaves.insert(*id, parent_id);
                }
            }
        }
    }

    pub fn grow_leaves(&mut self, _engine: &mut GraphEngine) {
        self.pruned_leaves.clear();
    }

    pub fn is_finished(&self, engine: &GraphEngine) -> bool {
        let min_temp = match engine.quality {
            LayoutQuality::Draft => 0.1,
            LayoutQuality::Default => MIN_TEMPERATURE,
            LayoutQuality::Proof => 0.001,
        };
        self.temp < min_temp && self.pruned_leaves.is_empty()
    }
}

/// Finds the depth of the Lowest Common Ancestor (LCA) of two nodes.
fn calculate_lca_depth(
    id1: NodeId,
    id2: NodeId,
    nodes: &[crate::graph_engine::GraphNode],
) -> usize {
    let n1 = &nodes[id1.0];
    let n2 = &nodes[id2.0];

    let mut d1 = n1.depth;
    let mut d2 = n2.depth;

    // 1. Level the depths
    let mut curr1 = id1;
    let mut curr2 = id2;

    while d1 > d2 {
        if let Some(p) = nodes[curr1.0].parent {
            curr1 = p;
            d1 -= 1;
        } else {
            break;
        }
    }

    while d2 > d1 {
        if let Some(p) = nodes[curr2.0].parent {
            curr2 = p;
            d2 -= 1;
        } else {
            break;
        }
    }

    // 2. Walk up in lock-step
    while curr1 != curr2 {
        if let (Some(p1), Some(p2)) = (nodes[curr1.0].parent, nodes[curr2.0].parent) {
            curr1 = p1;
            curr2 = p2;
            d1 -= 1;
        } else {
            return 0; // Root is the LCA
        }
    }

    d1
}

/// Walks up the parent tree to find the effective "proxy" node for the current layout phase.
/// If we are laying out a specific directory, it clamps to the boundary of that directory.
fn resolve_proxy_node(
    start_id: NodeId,
    nodes: &[crate::graph_engine::GraphNode],
    participating_nodes: &HashSet<NodeId>,
) -> NodeId {
    let mut current = start_id;

    // Walk up the tree as long as the parent exists
    while let Some(parent_id) = nodes[current.0].parent {
        // If the current node is already one of the top-level nodes in this layout pass, stop.
        if participating_nodes.contains(&current) {
            break;
        }
        current = parent_id;
    }

    // Fallback to the highest ancestor we found that is part of the layout
    if participating_nodes.contains(&current) {
        current
    } else {
        start_id
    }
}
