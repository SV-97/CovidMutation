use std::fmt;
use std::ops::{Add, Mul, MulAssign, Shr};

use itertools::{EitherOrBoth, Itertools};
use num::Num;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Doesn't normalize (trim trailing zeros etc.)
pub struct Polynomial<F> {
    pub coeffs: Vec<F>,
}

impl<F: Clone> Polynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Polynomial { coeffs }
    }

    pub fn zero() -> Self {
        Polynomial { coeffs: vec![] }
    }
}

impl<F: Num + PartialEq + fmt::Display> fmt::Display for Polynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
        let mut it = self
            .coeffs
            .iter()
            .enumerate()
            .filter(|(i, c)| c != &&F::zero());
        let first = it.next().unwrap();
        let write_polynomial = |f: &mut fmt::Formatter<'_>, (degree, coeff)| -> fmt::Result {
            if degree == 0 {
                return write!(f, "{}", coeff);
            } else if coeff == &F::one() {
                write!(f, "x")?;
            } else {
                write!(f, "{}⋅x", coeff)?;
            }
            if degree != 1 {
                for n in (0..=(degree as f64).log10().floor() as u32)
                    .into_iter()
                    .rev()
                {
                    let m = (degree / (10_usize).pow(n)) % 10;
                    write!(f, "{}", table[m])?;
                }
            }
            fmt::Result::Ok(())
        };
        write_polynomial(f, first)?;
        for (i, c) in it {
            write!(f, "+")?;
            write_polynomial(f, (i, c))?;
        }
        fmt::Result::Ok(())
    }
}

impl<F: Num + Clone> Shr<usize> for Polynomial<F> {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        let mut c = vec![F::zero(); rhs];
        c.extend(self.coeffs.into_iter());
        Polynomial::new(c)
    }
}

impl<F: Num + Clone> Polynomial<F> {
    pub fn left_pad_zeros(self, i: usize) -> Self {
        self >> i
    }
}

impl<F: Num + Clone> Add for Polynomial<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let coeffs = self
            .coeffs
            .into_iter()
            .zip_longest(rhs.coeffs.into_iter())
            .map(|b| match b {
                EitherOrBoth::Both(x, y) => x + y,
                EitherOrBoth::Left(x) => x,
                EitherOrBoth::Right(y) => y,
            })
            .collect();
        Polynomial::new(coeffs)
    }
}

impl<F: Num + Clone> Mul<F> for Polynomial<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        Polynomial::new(self.coeffs.into_iter().map(|c| rhs.clone() * c).collect())
    }
}

impl<F: Num + Copy> Mul<&Polynomial<F>> for Polynomial<F> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(i, coeff)| (rhs.clone() * (*coeff)).left_pad_zeros(i))
            .fold(Self::zero(), Add::add)
    }
}

impl<F: Num + Copy> Mul<Polynomial<F>> for Polynomial<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<F: Num + Clone> MulAssign<&Polynomial<F>> for Polynomial<F> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, coeff)| (rhs.clone() * coeff.clone()).left_pad_zeros(i))
            .fold(Self::zero(), Add::add);
    }
}

impl<F> Polynomial<F>
where
    F: Num + Clone,
{
    pub fn one() -> Self {
        Polynomial {
            coeffs: vec![F::one()],
        }
    }

    pub fn powi(&self, i: usize) -> Self {
        match i {
            0 => Self::one(),
            1 => self.clone(),
            _ => {
                // exponentiation by squaring as described in
                let mut res = self.clone();
                for c in format!("{:b}", i).chars() {
                    match c {
                        '0' => {
                            res *= &res.clone();
                        }
                        '1' => {
                            res *= &res.clone();
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
let p1: polynomial::Polynomial<isize> = polynomial::Polynomial::new(vec![-3, 2]);
let p2: polynomial::Polynomial<isize> = polynomial::Polynomial::new(vec![3, 2]);
println!("{}", p1 * p2);
let p3 = polynomial::Polynomial::new(vec![BigUint::one(); 10]);
println!("{}", p3.powi(10));
*/
