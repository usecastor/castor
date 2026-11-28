//! KV-fill priority ranking, soft pool limits, and admission logic.

use std::cmp::Ordering;

use super::lease::Lease;

/// Compare two leases by priority:
/// 1. Higher `context_tokens` is heavier, prioritized to finish and free its large memory.
/// 2. Older task (smaller timestamp `at`) is senior.
/// 3. Deterministic tie-breaker by `task_id`.
pub fn lease_rank(a: &Lease, b: &Lease) -> Ordering {
    // 1. Higher context_tokens is heavier, prioritized to finish and free its large memory
    if a.context_tokens != b.context_tokens {
        return a.context_tokens.cmp(&b.context_tokens);
    }
    // 2. Older task (smaller timestamp `at`) is senior
    if a.at != b.at {
        return b.at.cmp(&a.at);
    }
    // 3. Deterministic tie-breaker
    b.task_id.cmp(&a.task_id)
}

/// Soft limit for the pool's token budget (90% of capacity).
pub fn soft_pool_limit(pool_capacity_tokens: u32) -> u64 {
    (pool_capacity_tokens as u64 * 90) / 100
}

/// Total committed context tokens across a slice of active leases,
/// charging `per_slot_state_tokens` overhead for each active slot.
pub fn committed_tokens_with_overhead(leases: &[Lease], per_slot_state_tokens: u32) -> u64 {
    leases
        .iter()
        .map(|l| (l.context_tokens as u64).saturating_add(per_slot_state_tokens as u64))
        .sum()
}

/// Total committed context tokens across a slice of active leases.
pub fn committed_tokens(leases: &[Lease]) -> u64 {
    committed_tokens_with_overhead(leases, 0)
}

/// Whether the aggregate KV pool is exhausted (committed >= capacity).
pub fn is_pool_exhausted(pool_capacity_tokens: u32, committed: u64) -> bool {
    if pool_capacity_tokens == 0 {
        return false;
    }
    committed >= pool_capacity_tokens as u64
}

/// DRY token-budget admission gate with per-slot state overhead.
///
/// Halts new task admission if aggregate committed tokens exceed the soft limit,
/// or if the new task's initial tokens plus its per-slot state overhead would
/// push the pool over total capacity.
pub fn token_budget_admits_with_overhead(
    pool_capacity_tokens: u32,
    committed: u64,
    initial_tokens: u32,
    per_slot_state_tokens: u32,
) -> bool {
    if pool_capacity_tokens == 0 {
        return true; // token-budget gate disabled
    }
    let soft_limit = soft_pool_limit(pool_capacity_tokens);
    if committed > soft_limit {
        return false;
    }
    let new_total = committed
        .saturating_add(initial_tokens as u64)
        .saturating_add(per_slot_state_tokens as u64);
    new_total <= pool_capacity_tokens as u64
}

/// DRY token-budget admission gate.
///
/// Halts new task admission if aggregate committed tokens exceed the soft limit,
/// or if the new task's initial tokens would push the pool over total capacity.
pub fn token_budget_admits(pool_capacity_tokens: u32, committed: u64, initial_tokens: u32) -> bool {
    token_budget_admits_with_overhead(pool_capacity_tokens, committed, initial_tokens, 0)
}

/// Decide whether an active lease should yield (pause) to allow
/// a heavier or senior active task to finish and free its KV cache,
/// accounting for `per_slot_state_tokens` per active slot.
pub fn should_yield_with_overhead(
    slots: &[Lease],
    task_id: &str,
    tenant: u32,
    pool_capacity_tokens: u32,
    per_slot_state_tokens: u32,
) -> bool {
    if slots.len() <= 1 {
        // Solo worker: no one to yield to, and no contention.
        return false;
    }
    if pool_capacity_tokens == 0 {
        return false;
    }
    let committed = committed_tokens_with_overhead(slots, per_slot_state_tokens);
    if committed <= soft_pool_limit(pool_capacity_tokens) {
        return false;
    }

    let our_lease = slots
        .iter()
        .find(|l| l.task_id == task_id && l.tenant == tenant);
    let Some(ours) = our_lease else {
        return false;
    };

    // If there exists ANY other active lease with higher rank (heavier context or senior),
    // we yield (pause) so that task can finish and release its KV footprint.
    slots
        .iter()
        .any(|other| other.task_id != ours.task_id && lease_rank(other, ours) == Ordering::Greater)
}

/// Decide whether an active lease should yield (pause) to allow
/// a heavier or senior active task to finish and free its KV cache.
pub fn should_yield(
    slots: &[Lease],
    task_id: &str,
    tenant: u32,
    pool_capacity_tokens: u32,
) -> bool {
    should_yield_with_overhead(slots, task_id, tenant, pool_capacity_tokens, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(task_id: &str, tenant: u32, at: u64, context_tokens: u32) -> Lease {
        Lease {
            tenant,
            task_id: task_id.to_string(),
            at,
            hb: at,
            context_tokens,
        }
    }

    #[test]
    fn committed_tokens_includes_per_slot_overhead() {
        let slots = vec![lease("a", 1, 10, 30), lease("b", 1, 20, 40)];
        assert_eq!(committed_tokens(&slots), 70);
        assert_eq!(committed_tokens_with_overhead(&slots, 15), 100);
    }

    #[test]
    fn admission_checks_initial_plus_overhead_against_capacity() {
        // Pool 100, soft limit 90. Committed 50 <= 90, adding 30 + 20 = 100 <= 100 -> admit.
        assert!(token_budget_admits_with_overhead(100, 50, 30, 20));
        // Adding 31 + 20 = 101 > 100 -> reject.
        assert!(!token_budget_admits_with_overhead(100, 50, 31, 20));
        // Committed 91 > soft limit 90 -> reject even with 0 initial/overhead.
        assert!(!token_budget_admits_with_overhead(100, 91, 0, 0));
    }

    #[test]
    fn should_yield_triggers_when_overhead_pushes_committed_past_soft_limit() {
        let slots = vec![lease("heavy", 1, 10, 40), lease("light", 1, 20, 30)];
        // Raw committed = 70 <= 90 -> no yield with 0 overhead.
        assert!(!should_yield_with_overhead(&slots, "light", 1, 100, 0));
        // With 15 overhead per slot: committed = 70 + 30 = 100 > 90 -> light yields to heavy.
        assert!(should_yield_with_overhead(&slots, "light", 1, 100, 15));
        assert!(!should_yield_with_overhead(&slots, "heavy", 1, 100, 15));
    }
}
