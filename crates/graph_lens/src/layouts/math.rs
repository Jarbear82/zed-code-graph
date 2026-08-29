/// Matrix representation optimized for graph layout operations.
/// Stores data in row-major order.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Set element at (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    /// Multiplies the matrix by a vector: y = A * x
    /// x must have length equal to self.cols.
    /// Returns a vector of length self.rows.
    pub fn multiply_vec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.cols,
            "Vector length must match matrix columns"
        );
        let mut y = vec![0.0; self.rows];

        for (i, y_val) in y.iter_mut().enumerate() {
            let row_offset = i * self.cols;
            let mut sum = 0.0;
            for (j, x_val) in x.iter().enumerate() {
                sum += self.data[row_offset + j] * x_val;
            }
            *y_val = sum;
        }

        y
    }
}

/// Computes the dot product of two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector lengths must match for dot product"
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the L2 norm (magnitude) of a vector.
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalizes a vector in-place. Returns the original norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let n = norm(v);
    if n > 1e-10 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

/// Orthogonalizes vector `v` against `u` using Gram-Schmidt:
/// v = v - (v . u) * u
/// Assumes `u` is already normalized (unit vector).
pub fn orthogonalize_unit(v: &mut [f32], u: &[f32]) {
    assert_eq!(
        v.len(),
        u.len(),
        "Vector lengths must match for orthogonalization"
    );
    let projection = dot(v, u);
    for i in 0..v.len() {
        v[i] -= projection * u[i];
    }
}

/// Generates a random unit vector of length `n`.
pub fn random_unit_vector(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| rand::random::<f32>() - 0.5).collect();
    normalize(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_vec_mul() {
        let mut m = Matrix::new(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let x = vec![1.0, 1.0];
        let y = m.multiply_vec(&x);

        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_orthogonalize() {
        let mut v = vec![1.0, 1.0];
        let u = vec![1.0, 0.0]; // Unit vector on X axis

        orthogonalize_unit(&mut v, &u);

        assert!((v[0]).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }
}
