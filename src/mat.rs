use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, MulAssign, Neg, Sub};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Mat<const ROWS: usize, const COLS: usize> {
    rows: [[f64; COLS]; ROWS],
}

pub type Vector<const ROWS: usize> = Mat<ROWS, 1>;
pub type Covector<const COLS: usize> = Mat<1, COLS>;

impl<const I: usize, const J: usize> fmt::Display for Mat<I, J> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[ {}×{} Matrix", I, J)?;
        for row in self.rows.iter() {
            for val in row.iter() {
                write!(f, "{:3},", val)?
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl<const I: usize, const J: usize> Mat<I, J> {
    pub fn new(rows: [[f64; J]; I]) -> Self {
        Mat { rows }
    }

    pub fn zero() -> Self {
        Self::new([[0.; J]; I])
    }

    pub fn transpose(self) -> Mat<J, I> {
        let mut m = Mat::zero();
        for i in 0..I {
            for j in 0..J {
                m[(j, i)] = self[(i, j)];
            }
        }
        m
    }

    pub fn row(&self, i: usize) -> Mat<1, J> {
        Mat {
            rows: [self.rows[i]],
        }
    }

    pub fn set_row(&mut self, i: usize, row: Mat<1, J>) {
        self.rows[i] = row.rows[0];
    }

    pub fn col(&self, j: usize) -> Mat<I, 1> {
        let mut v = Mat::zero();
        for i in 0..I {
            v[i] = self[(i, j)];
        }
        v
    }

    pub fn row_minor(&self, i: usize) -> Mat<{ I - 1 }, J> {
        //let m = [[f64; J]; I-1]::try_from(self.rows[0..I - 1])
        let mut m = Mat::zero();
        for row in 0..i {
            m.rows[row] = self.rows[row];
        }
        for row in i + 1..I {
            m.rows[row - 1] = self.rows[row];
        }
        m
    }

    pub fn col_minor(&self, j: usize) -> Mat<I, { J - 1 }> {
        self.transpose().row_minor(j).transpose()
    }

    pub fn minor(&self, i: usize, j: usize) -> Mat<{ I - 1 }, { J - 1 }> {
        self.row_minor(i).col_minor(j)
    }
}

impl<const I: usize> Mat<I, I> {
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for i in 0..I {
            m[(i, i)] = 1.0;
        }
        m
    }
    pub fn powi(&self, i: usize) -> Self {
        match i {
            0 => Self::identity(),
            1 => *self,
            _ => {
                // exponentiation by squaring as described in
                let mut res = *self;
                for c in format!("{:b}", i).chars() {
                    match c {
                        '0' => {
                            res *= res;
                        }
                        '1' => {
                            res *= res;
                            res *= self;
                        }
                        _ => unreachable!(),
                    }
                }
                res
            }
        }
    }
}

/*
impl<const I: usize> Mat<I, I>
where
    [(); I - 1]: ,
{
    pub fn det(&self) -> f64 {
        if I == 1 {
            self[0]
        } else {
            let mut d = 0.;
            for (i, x) in self.row(0).rows.iter().enumerate() {
                d += x[0] * (self.minor(0, i).det());
            }
            d
        }
    }
}
*/

impl<const I: usize> Vector<I> {
    pub fn new_vec(rows: [f64; I]) -> Self {
        Mat::new([rows]).transpose()
    }
}

impl<const I: usize> Covector<I> {
    pub fn new_covec(cols: [f64; I]) -> Self {
        Mat::new([cols])
    }

    pub fn dot(self, rhs: Mat<I, 1>) -> f64 {
        let mut dot_prod = 0.0;
        for i in 0..I {
            dot_prod += self[i] * rhs[i];
        }
        dot_prod
    }

    pub fn dot_ref(&self, rhs: &Mat<I, 1>) -> f64 {
        let mut dot_prod = 0.0;
        for i in 0..I {
            dot_prod += self[i] * rhs[i];
        }
        dot_prod
    }
}

impl<const I: usize, const J: usize> IndexMut<(usize, usize)> for Mat<I, J> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.rows[row][col]
    }
}

impl<const I: usize, const J: usize> IndexMut<usize> for Mat<I, J> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let j = index % J;
        let i = (index - j) / J;
        &mut self[(i, j)]
    }
}

impl<const I: usize, const J: usize> Index<(usize, usize)> for Mat<I, J> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.rows[row][col]
    }
}

impl<const I: usize, const J: usize> Index<usize> for Mat<I, J> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        let i = index / J;
        let j = index - J * i;
        &self[(i, j)]
    }
}

impl<const I: usize, const J: usize> Add for Mat<I, J> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut v = Self::zero();
        for i in 0..I {
            for j in 0..J {
                v[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }
        v
    }
}

impl<const I: usize, const J: usize> Sub for Mat<I, J> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut v = Self::zero();
        for i in 0..I {
            for j in 0..J {
                v[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }
        v
    }
}

impl<const I: usize, const J: usize> Neg for Mat<I, J> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut v = Self::zero();
        for i in 0..I {
            for j in 0..J {
                v[(i, j)] = -self[(i, j)];
            }
        }
        v
    }
}

impl<const I: usize, const J: usize, const N: usize> Mul<Mat<J, N>> for Mat<I, J> {
    type Output = Mat<I, N>;

    fn mul(self, rhs: Mat<J, N>) -> Self::Output {
        let mut m = Self::Output::zero();
        for i in 0..I {
            for j in 0..N {
                m[(i, j)] = self.row(i).dot(rhs.col(j));
            }
        }
        m
    }
}

impl<const I: usize, const J: usize> MulAssign<Mat<J, J>> for Mat<I, J> {
    fn mul_assign(&mut self, rhs: Mat<J, J>) {
        let mut m = Self::zero();
        for i in 0..I {
            for j in 0..J {
                m[(i, j)] = self.row(i).dot(rhs.col(j));
            }
        }
        *self = m;
    }
}

impl<const I: usize, const J: usize> MulAssign<&Mat<J, J>> for Mat<I, J> {
    fn mul_assign(&mut self, rhs: &Mat<J, J>) {
        let mut m = Self::zero();
        for i in 0..I {
            for j in 0..J {
                m[(i, j)] = self.row(i).dot_ref(&rhs.col(j));
            }
        }
        *self = m;
    }
}

impl<const I: usize, const J: usize> Mul<f64> for Mat<I, J> {
    type Output = Mat<I, J>;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut m = Self::Output::zero();
        for i in 0..I {
            for j in 0..J {
                m[(i, j)] = rhs * self[(i, j)];
            }
        }
        m
    }
}

impl<const I: usize, const J: usize> Mul<Mat<I, J>> for f64 {
    type Output = Mat<I, J>;

    fn mul(self, rhs: Mat<I, J>) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mat_ident() {
        assert_eq!(
            Mat::<4, 4>::identity(),
            Mat::<4, 4>::new([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ])
        );
    }

    #[test]
    fn mat_add() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let b = Mat::new([[-1., 2., -3.], [4., -5., 6.], [-7., 8., -9.]]);
        let c = Mat::new([[0., 4., 0.], [8., 0., 12.], [0., 16., 0.]]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn mat_neg() {
        let a = Mat::new([[1., 2.], [3., 4.]]);
        let b = Mat::new([[-1., -2.], [-3., -4.]]);
        assert_eq!(-a, b);
    }

    #[test]
    fn test_row() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]);
        let b = Covector::new_covec([4., 5., 6.]);
        assert_eq!(a.row(1), b);
    }

    #[test]
    fn test_col() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]);
        let b = Vector::new_vec([1., 4., 7., 10.]);
        assert_eq!(a.col(0), b);
    }

    #[test]
    fn test_dot() {
        assert_eq!(
            Covector::new_covec([1., 2., 3.]).dot(Vector::new_vec([-1., 4., 5.])),
            22.
        );
    }

    #[test]
    fn test_mul() {
        let a = Mat::new([
            [5., 10., 15.],
            [6., 36., 12.],
            [7., 14., -100.],
            [3., -4., 5.],
        ]);
        let b = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let ab = Mat::new([
            [150., 180., 210.],
            [234., 288., 342.],
            [-637., -716., -795.],
            [22., 26., 30.],
        ]);

        assert_eq!(a * b, ab);
        assert_eq!(a * b, (b.transpose() * a.transpose()).transpose());
    }

    #[test]
    fn row_minor() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(a.row_minor(0), Mat::new([[4., 5., 6.], [7., 8., 9.]]));
    }

    #[test]
    fn col_minor() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(a.col_minor(1), Mat::new([[1., 3.], [4., 6.], [7., 9.]]));
    }

    #[test]
    fn minor() {
        let a = Mat::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(a.minor(2, 2), Mat::new([[1., 2.], [4., 5.]]));
    }

    #[test]
    fn test_det() {}
}
