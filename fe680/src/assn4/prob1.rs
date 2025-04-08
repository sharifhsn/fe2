use polars::prelude::*;

pub fn a() -> PolarsResult<()> {
    // Table 1: Unconditional default probabilities
    let table1 = df![
        "Time (years)" => &[1, 2, 3, 4, 5],
        "Default probability" => &[0.0200, 0.0196, 0.0192, 0.0188, 0.0184],
        "Survival probability" => &[0.9800, 0.9604, 0.9412, 0.9224, 0.9039],
    ]?;

    // Table 2: Present values of expected payments
    let table2 = df![
        "Time (years)" => &[1, 2, 3, 4, 5],
        "Survival probability" => &[0.9800, 0.9604, 0.9412, 0.9224, 0.9039],
        "Expected payment" => &[0.9800, 0.9604, 0.9412, 0.9224, 0.9039], // times s
        "Discount factor" => &[0.9512, 0.9048, 0.8607, 0.8187, 0.7788],
        "PV of expected payment" => &[0.9322, 0.8690, 0.8101, 0.7552, 0.7040], // times s
    ]?;

    // Table 3: Present value of expected payoff
    let table3 = df![
        "Time (years)" => &[0.5, 1.5, 2.5, 3.5, 4.5],
        "Probability of default" => &[0.0200, 0.0196, 0.0192, 0.0188, 0.0184],
        "Recovery rate" => &[0.4; 5],
        "Expected payoff ($)" => &[0.0120, 0.0118, 0.0115, 0.0113, 0.0111],
        "Discount factor" => &[0.9753, 0.9277, 0.8825, 0.8395, 0.7985],
        "PV of expected payoff ($)" => &[0.0117, 0.0109, 0.0102, 0.0095, 0.0088],
    ]?;

    // Table 4: Present value of accrual payment
    let table4 = df![
        "Time (years)" => &[0.5, 1.5, 2.5, 3.5, 4.5],
        "Probability of default" => &[0.0200, 0.0196, 0.0192, 0.0188, 0.0184],
        "Expected accrual payment" => &[0.0100, 0.0098, 0.0096, 0.0094, 0.0092], // times s
        "Discount factor" => &[0.9753, 0.9277, 0.8825, 0.8395, 0.7985],
        "PV of expected accrual payment" => &[0.0097, 0.0091, 0.0085, 0.0079, 0.0074], // times s
    ]?;

    Ok(())
}
