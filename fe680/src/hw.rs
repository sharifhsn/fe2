// fn p_u(&self, branch_mode: BranchMode, j: f64) -> f64 {
//     1.0 / 6.0 + 0.5 * (self.a.powi(2) * j.powi(2) * self.dt.powi(2) - self.a * j * self.dt) +
//     match branch_mode {
//         BranchMode::Middle => 0.0,
//         BranchMode::Up => self.a * j * self.dt,
//         BranchMode::Down => 1.0 - self.a * j * self.dt,
//     }
// }

// fn p_m(&self, branch_mode: BranchMode, j: f64) -> f64 {
//     2.0 / 3.0 - self.a.powi(2) * j.powi(2) * self.dt.powi(2) +
//     match branch_mode {
//         BranchMode::Middle => 0.0,
//         BranchMode::Up => -1.0 - 2.0 * self.a * j * self.dt,
//         BranchMode::Down => -1.0 + 2.0 * self.a * j * self.dt,
//     }
// }

// fn p_d(&self, branch_mode: BranchMode, j: f64) -> f64 {
//     1.0 / 6.0 + 0.5 * (self.a.powi(2) * j.powi(2) * self.dt.powi(2) + self.a * j * self.dt) +
//     match branch_mode {
//         BranchMode::Middle => 0.0,
//         BranchMode::Up => 1.0 + self.a * j * self.dt,
//         BranchMode::Down => -self.a * j * self.dt,
//     }
// }
