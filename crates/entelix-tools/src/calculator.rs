//! `Calculator` — arithmetic over `f64`.
//!
//! Recursive-descent parser supporting:
//!
//! - Numbers (integer and decimal).
//! - Binary `+ - * /` with the usual precedence.
//! - Unary `+` / `-`.
//! - Parentheses.
//! - Exponentiation `^` (right-associative).
//!
//! Out of scope: variables, function calls (`sin`, `sqrt`, …), `eval`-style
//! injection. Anything outside the grammar above is rejected with a
//! [`ToolError::Calculator`] message pinpointing the offending span.
//!
//! The parser is small enough to audit at a glance and depends on no
//! third-party expression library — security-relevant since this
//! tool evaluates LLM-generated input.

use serde::Serialize;

use entelix_core::AgentContext;
use entelix_core::error::Result;
use entelix_tool_derive::tool;

use crate::error::ToolError;

/// Typed output of [`calculator`] — surfaces both the original
/// expression (for round-trip diagnostics) and the evaluated `f64`.
#[derive(Debug, Serialize)]
pub struct CalculatorOutput {
    /// The expression that produced [`Self::result`].
    pub expression: String,
    /// Evaluated result (`f64`).
    pub result: f64,
}

#[tool(effect = "ReadOnly", idempotent)]
/// Evaluate an arithmetic expression. Supports `+ - * / ^`, unary minus, and parentheses; no variables or named functions. Returns the `f64` result.
#[allow(clippy::unused_async)] // `#[tool]` requires `async fn`; the body has no `.await` because the parser is synchronous.
pub async fn calculator(_ctx: &AgentContext<()>, expression: String) -> Result<CalculatorOutput> {
    let result = evaluate(&expression).map_err(ToolError::Calculator)?;
    Ok(CalculatorOutput { expression, result })
}

// ── Parser ─────────────────────────────────────────────────────────

/// Maximum allowed parenthesis-nesting depth. Generous for any
/// human-authored expression, bounds adversarial nesting from LLM
/// output that would otherwise blow the parser stack.
const MAX_PAREN_DEPTH: usize = 64;

/// Maximum tokens accepted from a single input. Caps quadratic-ish
/// behavior on pathologically large inputs and rejects them up front
/// rather than after the parser walks them.
const MAX_TOKENS: usize = 4096;

/// Evaluate the expression. Returns the parser's error string on
/// rejection (caller wraps in `ToolError::Calculator`).
fn evaluate(input: &str) -> std::result::Result<f64, String> {
    let tokens = tokenize(input)?;
    if tokens.len() > MAX_TOKENS {
        return Err(format!(
            "input has {} tokens; limit is {MAX_TOKENS}",
            tokens.len()
        ));
    }
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let result = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        let pos = parser.pos;
        let tok = parser.tokens.get(pos).cloned().unwrap_or(Token::Plus);
        return Err(format!("unexpected token at position {pos}: '{tok:?}'"));
    }
    if !result.is_finite() {
        return Err(format!("result {result} is not finite"));
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn tokenize(input: &str) -> std::result::Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let mut buf = String::new();
            while let Some(&c2) = chars.peek() {
                if c2.is_ascii_digit() || c2 == '.' {
                    buf.push(c2);
                    chars.next();
                } else {
                    break;
                }
            }
            let num: f64 = buf
                .parse()
                .map_err(|_| format!("malformed number: '{buf}'"))?;
            out.push(Token::Num(num));
            continue;
        }
        let tok = match c {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '^' => Token::Caret,
            '(' => Token::LParen,
            ')' => Token::RParen,
            other => return Err(format!("unexpected character '{other}'")),
        };
        out.push(tok);
        chars.next();
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn advance(&mut self) {
        self.pos += 1;
    }
    fn parse_expr(&mut self) -> std::result::Result<f64, String> {
        let mut acc = self.parse_term()?;
        while let Some(t) = self.peek() {
            match t {
                Token::Plus => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    acc += rhs;
                }
                Token::Minus => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    acc -= rhs;
                }
                _ => break,
            }
        }
        Ok(acc)
    }
    fn parse_term(&mut self) -> std::result::Result<f64, String> {
        let mut acc = self.parse_unary()?;
        while let Some(t) = self.peek() {
            match t {
                Token::Star => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    acc *= rhs;
                }
                Token::Slash => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    if rhs == 0.0 {
                        return Err("division by zero".to_owned());
                    }
                    acc /= rhs;
                }
                _ => break,
            }
        }
        Ok(acc)
    }
    fn parse_unary(&mut self) -> std::result::Result<f64, String> {
        match self.peek() {
            Some(Token::Plus) => {
                self.advance();
                self.parse_unary()
            }
            Some(Token::Minus) => {
                self.advance();
                let v = self.parse_unary()?;
                Ok(-v)
            }
            _ => self.parse_pow(),
        }
    }
    fn parse_pow(&mut self) -> std::result::Result<f64, String> {
        let base = self.parse_atom()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.advance();
            let exp = self.parse_unary()?;
            Ok(base.powf(exp))
        } else {
            Ok(base)
        }
    }
    fn parse_atom(&mut self) -> std::result::Result<f64, String> {
        match self.peek() {
            Some(Token::Num(n)) => {
                let v = *n;
                self.advance();
                Ok(v)
            }
            Some(Token::LParen) => {
                self.advance();
                self.depth += 1;
                if self.depth > MAX_PAREN_DEPTH {
                    return Err(format!(
                        "parenthesis nesting exceeds limit ({MAX_PAREN_DEPTH})"
                    ));
                }
                let inner = self.parse_expr()?;
                if !matches!(self.peek(), Some(Token::RParen)) {
                    return Err("expected ')'".to_owned());
                }
                self.advance();
                self.depth -= 1;
                Ok(inner)
            }
            Some(other) => Err(format!("unexpected token '{other:?}'")),
            None => Err("unexpected end of input".to_owned()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests {
    use serde_json::json;

    use entelix_core::tools::Tool;

    use super::*;
    use crate::SchemaToolExt;

    fn ok(expr: &str, expected: f64) {
        let v = evaluate(expr).unwrap();
        assert!(
            (v - expected).abs() < 1e-9,
            "evaluate({expr}) = {v}, expected {expected}"
        );
    }
    fn err(expr: &str, contains: &str) {
        let e = evaluate(expr).unwrap_err();
        assert!(e.contains(contains), "expected '{contains}' in '{e}'");
    }

    #[test]
    fn simple_arithmetic() {
        ok("1 + 2", 3.0);
        ok("2 * 3 + 4", 10.0);
        ok("(2 + 3) * 4", 20.0);
        ok("10 / 4", 2.5);
        ok("2 ^ 3", 8.0);
        ok("-3 + 5", 2.0);
        ok("2 ^ 2 ^ 3", 256.0); // right-associative
    }

    #[test]
    fn rejects_division_by_zero() {
        err("1 / 0", "division by zero");
    }

    #[test]
    fn rejects_unknown_character() {
        err("1 + abc", "unexpected character");
    }

    #[tokio::test]
    async fn tool_execute_returns_result_envelope() {
        let adapter = Calculator.into_adapter();
        let out = adapter
            .execute(
                json!({"expression": "(2 + 3) * 4"}),
                &AgentContext::default(),
            )
            .await
            .unwrap();
        assert_eq!(out["expression"], "(2 + 3) * 4");
        assert_eq!(out["result"], 20.0);
    }

    #[test]
    fn metadata_carries_effect_and_idempotent_overrides() {
        let adapter = Calculator.into_adapter();
        let meta = Tool::metadata(&adapter);
        assert_eq!(meta.name, "calculator");
        assert!(matches!(
            meta.effect,
            entelix_core::tools::ToolEffect::ReadOnly
        ));
        assert!(meta.idempotent);
    }

    #[test]
    fn deep_paren_nesting_rejected_before_stack_blowout() {
        let expr = format!(
            "{open}1{close}",
            open = "(".repeat(100),
            close = ")".repeat(100)
        );
        err(&expr, "nesting exceeds limit");
    }

    #[test]
    fn nesting_at_the_limit_is_accepted() {
        let expr = format!(
            "{open}1{close}",
            open = "(".repeat(MAX_PAREN_DEPTH),
            close = ")".repeat(MAX_PAREN_DEPTH)
        );
        ok(&expr, 1.0);
    }

    #[test]
    fn overflow_to_infinity_rejected() {
        err("2 ^ 1024", "not finite");
    }

    #[test]
    fn nan_result_rejected() {
        err("(-1) ^ 0.5", "not finite");
    }

    #[test]
    fn token_limit_rejects_huge_inputs() {
        let expr = std::iter::repeat_n("1", 5_000)
            .collect::<Vec<_>>()
            .join("+");
        err(&expr, "limit is");
    }

    #[tokio::test]
    async fn tool_execute_rejects_malformed_input() {
        let adapter = Calculator.into_adapter();
        let err = adapter
            .execute(json!({"expression": "1 + abc"}), &AgentContext::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("unexpected character"));
    }
}
