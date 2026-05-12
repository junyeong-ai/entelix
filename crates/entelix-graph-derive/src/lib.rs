//! # entelix-graph-derive
//!
//! `#[derive(StateMerge)]` proc-macro for `StateGraph<S>` state
//! types. The derive emits two related items:
//!
//! 1. A `<Name>Contribution` companion struct where every field
//!    is `Option`-wrapped — the natural Rust shape for the
//!    `LangGraph` `TypedDict` "node returned only these slots"
//!    semantic.
//! 2. The `entelix_graph::StateMerge` impl on the input struct.
//!    `merge` combines two same-shape `S` values via per-field
//!    reducers (`Annotated<T, R>` fields call `R::reduce`; plain
//!    fields are replaced by `update`). `merge_contribution`
//!    folds an `Option`-wrapped contribution into the current
//!    state — fields the node *didn't write* leave the current
//!    value untouched, and fields it *did* write merge through
//!    the same per-field reducer.
//!
//! The companion struct also gets a builder-style `with_<field>`
//! method per field — for `Annotated<T, R>` fields the builder
//! takes raw `T` and wraps it with `R::default()` automatically,
//! so node bodies write `contribution.with_log(vec!["…"])`
//! rather than `contribution.with_log(Annotated::new(vec, Append::new()))`.
//!
//! ## Field detection
//!
//! - Field type is `Annotated<T, R>` (any path ending in
//!   `Annotated<…>` with at least one type argument): the
//!   contribution slot is `Option<Annotated<T, R>>`, the
//!   `merge`/`merge_contribution` paths call the bundled
//!   reducer, the builder accepts raw `T`.
//! - Any other type: the contribution slot is `Option<T>`, the
//!   `merge` path replaces, the builder accepts `T` directly.
//!
//! ## Generated impl (illustrative)
//!
//! For:
//!
//! ```ignore
//! #[derive(StateMerge, Clone, Default)]
//! struct AgentState {
//!     log: Annotated<Vec<String>, Append<String>>,
//!     score: Annotated<i32, Max<i32>>,
//!     last_message: String,
//! }
//! ```
//!
//! the macro emits (roughly):
//!
//! ```text
//! #[derive(Default)]
//! pub struct AgentStateContribution {
//!     pub log: Option<Annotated<Vec<String>, Append<String>>>,
//!     pub score: Option<Annotated<i32, Max<i32>>>,
//!     pub last_message: Option<String>,
//! }
//!
//! impl AgentStateContribution {
//!     pub fn with_log(mut self, v: Vec<String>) -> Self { ... }
//!     pub fn with_score(mut self, v: i32) -> Self { ... }
//!     pub fn with_last_message(mut self, v: String) -> Self { ... }
//! }
//!
//! impl ::entelix_graph::StateMerge for AgentState {
//!     type Contribution = AgentStateContribution;
//!     fn merge(self, update: Self) -> Self { ... }
//!     fn merge_contribution(self, c: Self::Contribution) -> Self { ... }
//! }
//! ```
//!
//! Tuple and unit structs are rejected at compile time.
//! Reducers used in `Annotated<T, R>` fields must implement
//! `Default` (the contribution builder needs to construct the
//! `Annotated` instance from raw `T`); the four stock reducers
//! (`Replace`, `Append`, `MergeMap`, `Max`) all qualify, as do
//! any unit-struct user reducers. Stateful reducers requiring
//! configuration are out of scope for the derive — operators
//! using them implement `StateMerge` manually.

#![doc(html_root_url = "https://docs.rs/entelix-graph-derive/0.5.2")]
#![deny(missing_docs)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Data, DataStruct, DeriveInput, Field, Fields, GenericArgument, Ident, PathArguments, Type,
    parse_macro_input, spanned::Spanned,
};

/// Derive `entelix_graph::StateMerge` and generate the
/// `<Name>Contribution` companion struct.
#[proc_macro_derive(StateMerge)]
pub fn derive_state_merge(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    expand(&ast)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(ast: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Struct(DataStruct { fields, .. }) = &ast.data else {
        return Err(syn::Error::new(
            ast.span(),
            "#[derive(StateMerge)] only supports structs",
        ));
    };
    let Fields::Named(named) = fields else {
        return Err(syn::Error::new(
            fields.span(),
            "#[derive(StateMerge)] requires named fields",
        ));
    };

    let name = &ast.ident;
    let vis = &ast.vis;
    let contribution_ident = format_ident!("{}Contribution", name);
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let descriptors: Vec<FieldDescriptor> = named.named.iter().map(FieldDescriptor::from).collect();

    let companion_fields = descriptors.iter().map(FieldDescriptor::companion_field);
    let companion_builders = descriptors.iter().map(FieldDescriptor::companion_builder);
    let mergers = descriptors.iter().map(FieldDescriptor::merge_arm);
    let contribution_mergers = descriptors
        .iter()
        .map(FieldDescriptor::merge_contribution_arm);

    Ok(quote! {
        #[derive(Default)]
        #vis struct #contribution_ident #ty_generics #where_clause {
            #(#companion_fields),*
        }

        impl #impl_generics #contribution_ident #ty_generics #where_clause {
            #(#companion_builders)*
        }

        impl #impl_generics ::entelix_graph::StateMerge for #name #ty_generics #where_clause {
            type Contribution = #contribution_ident #ty_generics;

            fn merge(self, update: Self) -> Self {
                Self {
                    #(#mergers),*
                }
            }

            fn merge_contribution(self, contribution: Self::Contribution) -> Self {
                Self {
                    #(#contribution_mergers),*
                }
            }
        }
    })
}

/// Per-field metadata the macro derives once and reuses across
/// every emit site. Pre-computing the kind keeps the four
/// `quote!` callsites readable.
struct FieldDescriptor<'a> {
    ident: &'a Ident,
    ty: &'a Type,
    annotated_inner: Option<&'a Type>,
}

impl<'a> From<&'a Field> for FieldDescriptor<'a> {
    fn from(field: &'a Field) -> Self {
        // The named-fields branch in `expand` already gated
        // tuple/unit structs out — every field reaching here has
        // an ident. `unwrap_or_else` keeps the macro
        // panic-free if a future call path ever reaches this
        // without that guarantee.
        let ident = field
            .ident
            .as_ref()
            .unwrap_or_else(|| unreachable!("FieldDescriptor::from called on tuple/unit field"));
        Self {
            ident,
            ty: &field.ty,
            annotated_inner: annotated_first_arg(&field.ty),
        }
    }
}

impl FieldDescriptor<'_> {
    fn companion_field(&self) -> TokenStream2 {
        let ident = self.ident;
        let ty = self.ty;
        quote! { pub #ident: ::core::option::Option<#ty> }
    }

    fn companion_builder(&self) -> TokenStream2 {
        let ident = self.ident;
        let setter = format_ident!("with_{}", ident);
        self.annotated_inner.map_or_else(
            || {
                let ty = self.ty;
                quote! {
                    /// Set this slot in the contribution.
                    #[must_use]
                    pub fn #setter(mut self, value: #ty) -> Self {
                        self.#ident = ::core::option::Option::Some(value);
                        self
                    }
                }
            },
            |inner| {
                quote! {
                    /// Set this slot in the contribution. The supplied
                    /// value is automatically wrapped in `Annotated`
                    /// using the field's reducer type's `Default` impl.
                    #[must_use]
                    pub fn #setter(mut self, value: #inner) -> Self {
                        self.#ident = ::core::option::Option::Some(
                            ::entelix_graph::Annotated::new(
                                value,
                                ::core::default::Default::default(),
                            ),
                        );
                        self
                    }
                }
            },
        )
    }

    /// Per-field merger for `merge(Self, Self) -> Self` —
    /// `Annotated<T, R>` calls reducer, plain field replaces.
    fn merge_arm(&self) -> TokenStream2 {
        let ident = self.ident;
        if self.annotated_inner.is_some() {
            quote! { #ident: self.#ident.merge(update.#ident) }
        } else {
            quote! { #ident: update.#ident }
        }
    }

    /// Per-field merger for `merge_contribution(Self, Contribution) -> Self`
    /// — `None` means "node didn't write this slot, keep current
    /// value"; `Some(v)` means "merge through reducer for
    /// `Annotated`, replace for plain".
    fn merge_contribution_arm(&self) -> TokenStream2 {
        let ident = self.ident;
        if self.annotated_inner.is_some() {
            quote! {
                #ident: match contribution.#ident {
                    ::core::option::Option::Some(v) => self.#ident.merge(v),
                    ::core::option::Option::None => self.#ident,
                }
            }
        } else {
            quote! {
                #ident: contribution.#ident.unwrap_or(self.#ident)
            }
        }
    }
}

/// `Some(inner)` when `ty` is syntactically `Annotated<inner, …>`
/// (any path length — `Annotated<…>`,
/// `entelix_graph::Annotated<…>`, or a user re-export like
/// `crate::state::Annotated<…>` all match). Returns the first
/// generic argument so the builder can take raw `T`. The macro
/// only inspects the *last* path segment; shadowing `Annotated`
/// elsewhere is forbidden by convention and would surface as a
/// type-check error during compilation of the generated code.
fn annotated_first_arg(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last = type_path.path.segments.last()?;
    if last.ident != Ident::new("Annotated", last.ident.span()) {
        return None;
    }
    let PathArguments::AngleBracketed(ref args) = last.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}
