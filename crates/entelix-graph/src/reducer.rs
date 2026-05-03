//! `Reducer<T>` — typed state-merge function.
//!
//! When two updates collide on the same state slot — typically because
//! two parallel branches each produced a value, or because a node's
//! return type only carries the *delta* it computed — a [`Reducer<T>`]
//! decides how to combine `(current, update)` into the next value.
//!
//! Each node returns a full state; reducers are standalone helpers
//! users call from their node closures, and the building block for
//! the field-level `Annotated<T, R>` merge convention.
//!
//! ## Stock impls
//!
//! - [`Replace`]   — last-write-wins (matches the current default).
//! - [`Append`]    — for `Vec<U>`; concatenates `current` and `update`.
//! - [`MergeMap`]  — for `HashMap<K, V>`; right-bias union.
//! - [`Max`]       — for any `T: Ord`; keeps the larger of the two.
//!
//! ## State-level composition
//!
//! [`StateMerge`] lifts the per-slot reducer pattern up to the
//! whole `S` shape: each implementor describes how an incoming
//! update folds into the current state. The companion
//! `entelix-graph-derive::StateMerge` proc-macro generates the
//! impl by walking struct fields — `Annotated<T, R>` fields apply
//! their bundled reducer; plain fields are replaced by the
//! incoming update.

use std::cmp::Ord;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// Combines a current value and an incoming update into the next value.
///
/// Implementors must be deterministic — the merge function runs
/// inside `CompiledGraph::execute_loop` and a reducer that depends on
/// outside state (random RNG, wall-clock time, …) breaks crash-resume
/// reproducibility.
pub trait Reducer<T>: Send + Sync + 'static {
    /// Combine `current` and `update`. Implementations are free to
    /// consume both; many simply return `update` (replace) or push
    /// the second into the first (append).
    fn reduce(&self, current: T, update: T) -> T;
}

/// Last-write-wins. Matches the current [`StateGraph`](crate::StateGraph)
/// default — included so users who explicitly opt into reducer
/// machinery have a no-op option.
#[derive(Clone, Copy, Debug, Default)]
pub struct Replace;

impl<T> Reducer<T> for Replace
where
    T: Send + Sync + 'static,
{
    fn reduce(&self, _current: T, update: T) -> T {
        update
    }
}

/// Append: `current` followed by `update`. Specialised for
/// `Vec<U>` where `U: Clone + Send + Sync + 'static`.
#[derive(Clone, Copy, Debug)]
pub struct Append<U>(PhantomData<fn() -> U>);

impl<U> Default for Append<U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<U> Append<U> {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<U> Reducer<Vec<U>> for Append<U>
where
    U: Send + Sync + 'static,
{
    fn reduce(&self, mut current: Vec<U>, mut update: Vec<U>) -> Vec<U> {
        current.append(&mut update);
        current
    }
}

/// Merge two `HashMap<K, V>`s, right-biased — entries from `update`
/// overwrite collisions in `current`. (Right-bias matches typical
/// LangGraph / dict-update semantics.)
#[derive(Clone, Copy, Debug)]
pub struct MergeMap<K, V>(PhantomData<fn() -> (K, V)>);

impl<K, V> Default for MergeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> MergeMap<K, V> {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<K, V> Reducer<HashMap<K, V>> for MergeMap<K, V>
where
    K: Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn reduce(&self, mut current: HashMap<K, V>, update: HashMap<K, V>) -> HashMap<K, V> {
        for (k, v) in update {
            current.insert(k, v);
        }
        current
    }
}

/// Keep the larger of `current` / `update` per `T: Ord`. Useful for
/// "highest score wins" reducers and `usize` step counters merged
/// across parallel branches.
#[derive(Clone, Copy, Debug)]
pub struct Max<T>(PhantomData<fn() -> T>);

impl<T> Default for Max<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Max<T> {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Reducer<T> for Max<T>
where
    T: Ord + Send + Sync + 'static,
{
    fn reduce(&self, current: T, update: T) -> T {
        std::cmp::max(current, update)
    }
}

/// Newtype that bundles a value `T` with the reducer `R` that should
/// merge it. The wrapper is a standalone helper users compose into
/// their state types; it does not hook into `StateGraph::add_node`
/// directly.
#[derive(Clone, Debug)]
pub struct Annotated<T, R>
where
    R: Reducer<T>,
{
    /// The current value.
    pub value: T,
    reducer: R,
}

impl<T, R> Default for Annotated<T, R>
where
    T: Default,
    R: Reducer<T> + Default,
{
    fn default() -> Self {
        Self {
            value: T::default(),
            reducer: R::default(),
        }
    }
}

impl<T, R> Annotated<T, R>
where
    R: Reducer<T>,
{
    /// Wrap `value` with the supplied reducer.
    pub const fn new(value: T, reducer: R) -> Self {
        Self { value, reducer }
    }

    /// Borrow the inner reducer.
    pub const fn reducer(&self) -> &R {
        &self.reducer
    }

    /// Consume the wrapper and return the inner value.
    pub fn into_value(self) -> T {
        self.value
    }

    /// Apply the bundled reducer to fold `update` into `self.value`.
    pub fn reduce_in_place(&mut self, update: T)
    where
        T: Default,
    {
        let current = std::mem::take(&mut self.value);
        self.value = self.reducer.reduce(current, update);
    }

    /// Consume `self`, return a new `Annotated<T, R>` with `update`
    /// folded in.
    #[must_use]
    pub fn reduced(self, update: T) -> Self
    where
        R: Clone,
    {
        let merged = self.reducer.reduce(self.value, update);
        Self {
            value: merged,
            reducer: self.reducer,
        }
    }

    /// Merge two annotated values. Both sides must agree on the
    /// reducer type by construction (they share `R`); the resulting
    /// `Annotated` keeps `self`'s reducer instance — this matters
    /// for stateful reducers, where the *current* slot's reducer
    /// has the right configuration. Stock reducers
    /// ([`Replace`], [`Append`], [`MergeMap`], [`Max`]) are unit
    /// structs, so the choice is moot for them.
    ///
    /// This is the building block the `StateMerge` derive macro
    /// emits per `Annotated<T, R>` field.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let merged = self.reducer.reduce(self.value, other.value);
        Self {
            value: merged,
            reducer: self.reducer,
        }
    }
}

/// State-level merge: how an incoming update folds into the current
/// state. The dispatch-loop counterpart to [`Reducer<T>`], one level
/// up — implemented on the whole `S` shape rather than a single
/// slot.
///
/// Two merge axes:
///
/// - [`Self::merge`] folds two same-shape `S` values (used by
///   parallel-branch joins via `add_send_edges`, where two
///   branches each produce a complete `S` and the dispatcher needs
///   to combine them).
/// - [`Self::merge_contribution`] folds an `Option`-wrapped
///   contribution from one node into the current state. The
///   `Contribution` type — declared via the `type Contribution`
///   associated item — names which slots the node *actually
///   wrote*, distinguishing "no contribution" from "contributed
///   the default value". This is the canonical
///   [`StateGraph::add_contributing_node`](crate::StateGraph::add_contributing_node)
///   entry point — closer to LangGraph's TypedDict
///   partial-return shape than the same-shape merge alone could
///   express.
///
/// The companion `entelix-graph-derive::StateMerge` derive macro
/// generates both methods plus the `<Name>Contribution` companion
/// struct. Manual impls are supported when field-by-field shape
/// doesn't fit (e.g. cross-field invariants enforced at merge time).
pub trait StateMerge: Sized {
    /// Companion type carrying an `Option`-wrapped slot per field
    /// of `Self`. The derive macro generates this struct
    /// automatically; manual implementors define their own.
    type Contribution: Default + Send + Sync + 'static;

    /// Fold `update` into `self` and return the merged state.
    /// Implementations must be deterministic for the same reason
    /// [`Reducer::reduce`] is — the merge runs inside the dispatch
    /// loop and a non-deterministic implementation breaks
    /// crash-resume reproducibility.
    #[must_use]
    fn merge(self, update: Self) -> Self;

    /// Fold a [`Self::Contribution`] (an `Option`-wrapped partial
    /// state) into `self`. Slots the node didn't write
    /// (`None`) leave the current value untouched; slots it did
    /// (`Some`) merge through the per-field reducer for
    /// [`Annotated<T, R>`] fields, or replace for plain fields.
    #[must_use]
    fn merge_contribution(self, contribution: Self::Contribution) -> Self;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn replace_returns_update() {
        let r = Replace;
        assert_eq!(r.reduce(1u32, 7), 7);
    }

    #[test]
    fn append_concatenates_vecs() {
        let r = Append::<u32>::new();
        assert_eq!(r.reduce(vec![1, 2], vec![3, 4]), vec![1, 2, 3, 4]);
    }

    #[test]
    fn append_handles_empty_inputs() {
        let r = Append::<u32>::new();
        assert_eq!(r.reduce(Vec::new(), vec![1]), vec![1]);
        assert_eq!(r.reduce(vec![1], Vec::new()), vec![1]);
        assert!(r.reduce(Vec::<u32>::new(), Vec::new()).is_empty());
    }

    #[test]
    fn merge_map_is_right_biased() {
        let r = MergeMap::<&str, i32>::new();
        let mut current = HashMap::new();
        current.insert("a", 1);
        current.insert("b", 2);
        let mut update = HashMap::new();
        update.insert("b", 20);
        update.insert("c", 3);
        let merged = r.reduce(current, update);
        assert_eq!(merged.get("a"), Some(&1));
        assert_eq!(merged.get("b"), Some(&20)); // right wins on collision
        assert_eq!(merged.get("c"), Some(&3));
    }

    #[test]
    fn max_keeps_larger() {
        let r = Max::<i32>::new();
        assert_eq!(r.reduce(3, 5), 5);
        assert_eq!(r.reduce(10, 5), 10);
        assert_eq!(r.reduce(-1, -3), -1);
    }

    #[test]
    fn annotated_reduced_returns_merged() {
        let a = Annotated::new(vec![1, 2], Append::<u32>::new());
        let b = a.reduced(vec![3]);
        assert_eq!(b.value, vec![1, 2, 3]);
    }

    #[test]
    fn annotated_reduce_in_place_updates_value() {
        let mut a = Annotated::new(vec![1, 2], Append::<u32>::new());
        a.reduce_in_place(vec![3, 4]);
        assert_eq!(a.value, vec![1, 2, 3, 4]);
    }

    #[test]
    fn annotated_into_value_unwraps() {
        let a = Annotated::new(42_i32, Replace);
        assert_eq!(a.into_value(), 42);
    }

    #[test]
    fn annotated_merge_combines_two_annotated_values() {
        let left = Annotated::new(vec![1u32, 2], Append::<u32>::new());
        let right = Annotated::new(vec![3u32, 4], Append::<u32>::new());
        let merged = left.merge(right);
        assert_eq!(merged.value, vec![1, 2, 3, 4]);
    }

    #[test]
    fn annotated_merge_respects_reducer_kind() {
        let left = Annotated::new(7_i32, Max::<i32>::new());
        let right = Annotated::new(4_i32, Max::<i32>::new());
        // Max-reducer keeps the larger of the two regardless of order.
        assert_eq!(left.merge(right).value, 7);
    }

    #[test]
    fn state_merge_can_be_implemented_manually() {
        // Hand-rolled impl confirms the trait surface is usable
        // without the derive macro — useful when a state type has
        // cross-field invariants that need merge-time enforcement.
        struct WithInvariant {
            log: Annotated<Vec<u32>, Append<u32>>,
            tag: String,
        }
        #[derive(Default)]
        struct WithInvariantContribution {
            log: Option<Annotated<Vec<u32>, Append<u32>>>,
            tag: Option<String>,
        }
        impl StateMerge for WithInvariant {
            type Contribution = WithInvariantContribution;
            fn merge(self, update: Self) -> Self {
                Self {
                    log: self.log.merge(update.log),
                    tag: update.tag,
                }
            }
            fn merge_contribution(self, c: Self::Contribution) -> Self {
                Self {
                    log: match c.log {
                        Some(v) => self.log.merge(v),
                        None => self.log,
                    },
                    tag: c.tag.unwrap_or(self.tag),
                }
            }
        }
        let merged = WithInvariant {
            log: Annotated::new(vec![1, 2], Append::new()),
            tag: "old".into(),
        }
        .merge(WithInvariant {
            log: Annotated::new(vec![3], Append::new()),
            tag: "new".into(),
        });
        assert_eq!(merged.log.value, vec![1, 2, 3]);
        assert_eq!(merged.tag, "new");

        // Contribution path: `tag` slot left None should keep the
        // current `tag`, `log` slot Some should merge through the
        // per-field reducer.
        let merged2 = WithInvariant {
            log: Annotated::new(vec![10], Append::new()),
            tag: "keep".into(),
        }
        .merge_contribution(WithInvariantContribution {
            log: Some(Annotated::new(vec![20], Append::new())),
            tag: None,
        });
        assert_eq!(merged2.log.value, vec![10, 20]);
        assert_eq!(merged2.tag, "keep");
    }

    #[test]
    fn reducer_object_is_dyn_safe() {
        // If this compiles, `Reducer<Vec<i32>>` is dyn-safe — useful
        // for users who want to swap reducers at runtime.
        let r: Box<dyn Reducer<Vec<i32>>> = Box::new(Append::<i32>::new());
        assert_eq!(r.reduce(vec![1], vec![2]), vec![1, 2]);
    }
}
