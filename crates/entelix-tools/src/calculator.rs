//! `CalculatorTool` — arithmetic over `f64`.
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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use entelix_core::context::ExecutionContext;
use entelix_core::error::Result;
use entelix_core::tools::{Tool, ToolEffect, ToolMetadata};

use crate::error::ToolError;

/// Calculator [`Tool`] for agentic workflows.
pub struct CalculatorTool {
    metadata: ToolMetadata,
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CalculatorTool {
    /// Build a calculator with the default declarative metadata.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: ToolMetadata::function(
                "calculator",
                "Evaluate an arithmetic expression. Supports + - * / ^ unary minus and \
                 parentheses; no variables or named functions. Returns the f64 result.",
                json!({
                    "type": "object",
                    "required": ["expression"],
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression (e.g. '2 + 3 * 4')."
                        }
                    }
                }),
            )
            .with_effect(ToolEffect::ReadOnly)
            .with_idempotent(true),
        }
    }
}

#[derive(Debug, Deserialize)]
struct CalcInput {
    expression: String,
}

#[derive(Debug, Serialize)]
struct CalcOutput {
    expression: String,
    result: f64,
}

#[async_trait]
impl Tool for CalculatorTool {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    async fn execute(&self, input: Value, _ctx: &ExecutionContext) -> Result<Value> {
        let parsed: CalcInput = serde_json::from_value(input).map_err(ToolError::from)?;
        let result = evaluate(&parsed.expression).map_err(ToolError::Calculator)?;
        let output = CalcOutput {
            expression: parsed.expression,
            result,
        };
        Ok(serde_json::to_value(output).map_err(ToolError::from)?)
    }
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
            "expression has {} tokens — limit is {MAX_TOKENS}",
            tokens.len()
        ));
    }
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
        depth: 0,
    };
    let value = parser.parse_expr()?;
    if parser.pos != tokens.len() {
        let trailing = tokens
            .get(parser.pos)
            .map_or_else(|| "<eof>".to_owned(), describe);
        return Err(format!(
            "trailing input after position {}: '{trailing}'",
            parser.pos
        ));
    }
    if !value.is_finite() {
        return Err(format!(
            "result is not finite ({}); operands likely overflow f64",
            if value.is_nan() {
                "NaN".to_owned()
            } else {
                value.to_string()
            }
        ));
    }
    Ok(value)
}

#[derive(Clone, Debug, PartialEq)]
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

fn describe(t: &Token) -> String {
    match t {
        Token::Num(n) => n.to_string(),
        Token::Plus => "+".into(),
        Token::Minus => "-".into(),
        Token::Star => "*".into(),
        Token::Slash => "/".into(),
        Token::Caret => "^".into(),
        Token::LParen => "(".into(),
        Token::RParen => ")".into(),
    }
}

fn tokenize(input: &str) -> std::result::Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut iter = input.chars().peekable();
    while let Some(&ch) = iter.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                iter.next();
            }
            '+' => {
                iter.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                iter.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                iter.next();
                tokens.push(Token::Star);
            }
            '/' => {
                iter.next();
                tokens.push(Token::Slash);
            }
            '^' => {
                iter.next();
                tokens.push(Token::Caret);
            }
            '(' => {
                iter.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                iter.next();
                tokens.push(Token::RParen);
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut buf = String::new();
                while let Some(&c) = iter.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        buf.push(c);
                        iter.next();
                    } else {
                        break;
                    }
                }
                let n: f64 = buf
                    .parse()
                    .map_err(|_| format!("invalid number literal '{buf}'"))?;
                tokens.push(Token::Num(n));
            }
            _ => return Err(format!("unexpected character '{ch}'")),
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    /// Open-parenthesis depth. Bumped on every `(`-driven recursion
    /// into `parse_expr` and decremented when the matching `)` is
    /// consumed. Capped at [`MAX_PAREN_DEPTH`].
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_term()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Plus => {
                    self.advance();
                    left += self.parse_term()?;
                }
                Token::Minus => {
                    self.advance();
                    left -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// term = factor (('*' | '/') factor)*
    fn parse_term(&mut self) -> std::result::Result<f64, String> {
        let mut left = self.parse_factor()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Star => {
                    self.advance();
                    left *= self.parse_factor()?;
                }
                Token::Slash => {
                    self.advance();
                    let rhs = self.parse_factor()?;
                    if rhs == 0.0 {
                        return Err("division by zero".into());
                    }
                    left /= rhs;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// factor = unary ('^' factor)?  -- right-associative ^
    fn parse_factor(&mut self) -> std::result::Result<f64, String> {
        let base = self.parse_unary()?;
        if let Some(Token::Caret) = self.peek() {
            self.advance();
            let exp = self.parse_factor()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    /// unary = ('+' | '-')* primary
    fn parse_unary(&mut self) -> std::result::Result<f64, String> {
        let mut sign = 1.0;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                }
                Some(Token::Minus) => {
                    self.advance();
                    sign = -sign;
                }
                _ => break,
            }
        }
        Ok(sign * self.parse_primary()?)
    }

    /// primary = NUM | '(' expr ')'
    fn parse_primary(&mut self) -> std::result::Result<f64, String> {
        match self.advance().cloned() {
            Some(Token::Num(n)) => Ok(n),
            Some(Token::LParen) => {
                if self.depth >= MAX_PAREN_DEPTH {
                    return Err(format!(
                        "parenthesis nesting exceeds limit of {MAX_PAREN_DEPTH}"
                    ));
                }
                self.depth += 1;
                let v = self.parse_expr()?;
                self.depth -= 1;
                match self.advance() {
                    Some(Token::RParen) => Ok(v),
                    Some(other) => Err(format!("expected ')', got '{}'", describe(other))),
                    None => Err("unclosed parenthesis".into()),
                }
            }
            Some(other) => Err(format!("unexpected token '{}'", describe(&other))),
            None => Err("unexpected end of input".into()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn ok(expr: &str, expected: f64) {
        let got = evaluate(expr).unwrap_or_else(|e| panic!("{expr} → {e}"));
        assert!(
            (got - expected).abs() < 1e-9,
            "{expr}: got {got}, expected {expected}"
        );
    }

    fn err(expr: &str, needle: &str) {
        let e = evaluate(expr).unwrap_err();
        assert!(
            e.contains(needle),
            "{expr}: error '{e}' did not contain '{needle}'"
        );
    }

    #[test]
    fn integer_addition() {
        ok("1 + 2", 3.0);
        ok("1+2+3+4", 10.0);
    }

    #[test]
    fn precedence() {
        ok("2 + 3 * 4", 14.0);
        ok("(2 + 3) * 4", 20.0);
        ok("10 - 2 * 3", 4.0);
    }

    #[test]
    fn unary_minus_chains() {
        ok("--5", 5.0);
        ok("-(-5)", 5.0);
        ok("-3 + 5", 2.0);
    }

    #[test]
    fn decimals() {
        ok("0.5 + 0.5", 1.0);
        ok("1.25 * 4", 5.0);
    }

    #[test]
    fn exponent_right_associative() {
        // 2^(3^2) = 2^9 = 512, not (2^3)^2 = 64.
        ok("2 ^ 3 ^ 2", 512.0);
    }

    #[test]
    fn division_by_zero_rejected() {
        err("1 / 0", "division by zero");
    }

    #[test]
    fn unknown_character_rejected() {
        err("1 + abc", "unexpected character 'a'");
    }

    #[test]
    fn unclosed_paren_rejected() {
        err("(1 + 2", "unclosed parenthesis");
    }

    #[test]
    fn empty_input_rejected() {
        err("", "unexpected end of input");
    }

    #[test]
    fn trailing_garbage_rejected() {
        err("1 + 2 3", "trailing input");
    }

    #[tokio::test]
    async fn tool_execute_returns_result_envelope() {
        let tool = CalculatorTool::new();
        let out = tool
            .execute(
                json!({"expression": "(2 + 3) * 4"}),
                &ExecutionContext::new(),
            )
            .await
            .unwrap();
        assert_eq!(out["expression"], "(2 + 3) * 4");
        assert_eq!(out["result"], 20.0);
    }

    #[test]
    fn deep_paren_nesting_rejected_before_stack_blowout() {
        // 100 nested `(` triggers the depth guard at 64.
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
        // 2^1024 saturates f64 to +inf — return InvalidRequest rather
        // than silently emitting Inf to the caller.
        err("2 ^ 1024", "not finite");
    }

    #[test]
    fn nan_result_rejected() {
        // 0^0 is well-defined as 1.0 in f64::powf, but 0/0 gets
        // caught by the divide-by-zero rule. Use a NaN-producing
        // pair: (-1)^0.5 → NaN.
        err("(-1) ^ 0.5", "not finite");
    }

    #[test]
    fn token_limit_rejects_huge_inputs() {
        // 5_000 plus-separated 1s — over the 4_096 token cap.
        let expr = std::iter::repeat_n("1", 5_000)
            .collect::<Vec<_>>()
            .join("+");
        err(&expr, "limit is");
    }

    #[tokio::test]
    async fn tool_execute_rejects_malformed_input() {
        let tool = CalculatorTool::new();
        let err = tool
            .execute(json!({"expression": "1 + abc"}), &ExecutionContext::new())
            .await
            .unwrap_err();
        // `Error::InvalidRequest` comes from the From impl in error.rs.
        assert!(format!("{err}").contains("unexpected character"));
    }
}
