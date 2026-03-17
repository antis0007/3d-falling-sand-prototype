use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct StartupGpuBudget {
    pub total_bytes_limit: u64,
    pub reserved_headroom_bytes: u64,
    pub planned_by_category: BTreeMap<&'static str, u64>,
    pub granted_by_category: BTreeMap<&'static str, u64>,
}

#[derive(Clone, Debug)]
pub struct StartupGpuBudgetDecision {
    pub budget: StartupGpuBudget,
    pub downgraded: bool,
}

#[derive(Clone, Debug)]
pub struct StartupGpuBudgetError {
    pub total_limit: u64,
    pub reserved_headroom: u64,
    pub planned_total: u64,
    pub granted_total: u64,
    pub details: String,
}

impl std::fmt::Display for StartupGpuBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "startup gpu budget exceeded: planned={}B granted={}B total_limit={}B reserved_headroom={}B details={}",
            self.planned_total,
            self.granted_total,
            self.total_limit,
            self.reserved_headroom,
            self.details,
        )
    }
}

impl std::error::Error for StartupGpuBudgetError {}

impl StartupGpuBudget {
    pub fn new(total_bytes_limit: u64, reserved_headroom_bytes: u64) -> Self {
        Self {
            total_bytes_limit,
            reserved_headroom_bytes,
            planned_by_category: BTreeMap::new(),
            granted_by_category: BTreeMap::new(),
        }
    }

    pub fn register_plan(&mut self, category: &'static str, planned_bytes: u64) {
        let entry = self.planned_by_category.entry(category).or_insert(0);
        *entry = entry.saturating_add(planned_bytes);
    }

    pub fn grant_planned(&mut self, category: &'static str) {
        let planned = self.planned_by_category.get(category).copied().unwrap_or(0);
        self.granted_by_category.insert(category, planned);
    }

    pub fn planned_total(&self) -> u64 {
        self.planned_by_category
            .values()
            .copied()
            .fold(0u64, |acc, value| acc.saturating_add(value))
    }

    pub fn granted_total(&self) -> u64 {
        self.granted_by_category
            .values()
            .copied()
            .fold(0u64, |acc, value| acc.saturating_add(value))
    }

    pub fn granted_budget_bytes(&self) -> u64 {
        self.total_bytes_limit
            .saturating_sub(self.reserved_headroom_bytes)
    }

    pub fn validate(&self) -> Result<(), StartupGpuBudgetError> {
        let planned_total = self.planned_total();
        let granted_total = self.granted_total();
        let granted_budget = self.granted_budget_bytes();
        if planned_total > granted_budget {
            return Err(StartupGpuBudgetError {
                total_limit: self.total_bytes_limit,
                reserved_headroom: self.reserved_headroom_bytes,
                planned_total,
                granted_total,
                details: format!(
                    "planned categories={:?} granted categories={:?}",
                    self.planned_by_category, self.granted_by_category
                ),
            });
        }
        Ok(())
    }
}

pub fn finalize_startup_budget(
    mut budget: StartupGpuBudget,
    min_headroom_bytes: u64,
) -> Result<StartupGpuBudgetDecision, StartupGpuBudgetError> {
    let mut downgraded = false;

    if budget.validate().is_err() {
        let downgraded_headroom = budget.reserved_headroom_bytes.max(min_headroom_bytes);
        if budget.reserved_headroom_bytes > min_headroom_bytes {
            budget.reserved_headroom_bytes = min_headroom_bytes;
            downgraded = true;
        }
        if let Err(mut err) = budget.validate() {
            err.details = format!(
                "{} (attempted_headroom_downgrade_to={}B from={}B)",
                err.details, min_headroom_bytes, downgraded_headroom
            );
            return Err(err);
        }
    }

    Ok(StartupGpuBudgetDecision { budget, downgraded })
}
