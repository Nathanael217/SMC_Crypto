"""
qf_smc/ai_verdict.py — AI verdict layer for SMC Long setups
============================================================
Builds prompt → calls LLM → parses JSON verdict.
Supports Groq (primary) and Anthropic (fallback).

Public API:
  - build_smc_prompt(...)
  - get_verdict(prompt, provider, api_key, model, timeout)
  - score_setup(result)
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List

import streamlit as st


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

PROMPT_TEMPLATE = """You are an analyst for the QUANTFLOW SMC Long system. Your job is to assess a long-only setup on a crypto altcoin perpetual and output a structured JSON verdict.

SETUP CONTEXT:

Symbol: {symbol}
Mode: {mode} (HTF={htf_label}, LTF={ltf_label})
BTC Regime: {btc_regime}{fights_macro_note}

MARKET STRUCTURE:
State: {state}
CHoCH bar index: {choch_bar}
BOS bar index: {bos_bar}
Current leg: from price ${leg_start_price:.6f} to ${leg_high_price:.6f}

ENTRY ZONE:
Primary zone type: {primary_zone_type}
Tier (if Smart OB): {ob_tier}
FVG status (if FVG): {fvg_status}
Price within Fibo 0.786 zone: {in_fibo_786}
At S/R support (if SR): touches={sr_touches}

CURRENT PRICE: ${current_price:.6f}

LTF CONFIRMATION: {ltf_confirmation}

BACKTEST EVIDENCE:
Per-coin (this coin's history): WR {wr_pc:.1%} | PF {pf_pc:.2f} | mean R {mr_pc:+.3f} (n={n_pc})
Universe baseline (top alts):    WR {wr_un:.1%} | PF {pf_un:.2f} | mean R {mr_un:+.3f} (n={n_un})
Bayesian blended:                WR {wr_bl:.1%} | PF {pf_bl:.2f} | mean R {mr_bl:+.3f} (n_eff={n_bl})

RECENT-vs-EARLIER CHECK:
Earlier period ({earlier_label}): mean R {mr_e:+.3f} | PF {pf_e:.2f} (n={n_e})
Recent period ({recent_label}):   mean R {mr_r:+.3f} | PF {pf_r:.2f} (n={n_r})
Verdict: {recent_verdict}

TRADE PLAN:
Entry: ${entry:.6f}
SL: ${sl:.6f}  (risk = {risk_pct:.2f}%)
TP1 (2R): ${tp1:.6f}
TP2 (2.5R): ${tp2:.6f}
TP3 (3R): ${tp3:.6f}
R:R to TP2: {rr:.2f}

YOUR TASK:
Assess this setup. Output ONLY a JSON object (no markdown, no preamble, no postamble):

{{
  "verdict": "TAKE" | "WAIT" | "SKIP",
  "confidence": 0-100,
  "reasoning": "<2-3 sentences explaining your reasoning>",
  "key_risks": ["<risk 1>", "<risk 2>"]
}}

DECISION FRAMEWORK:
- TAKE: blended PF >= 1.3, recent verdict not WEAKER, LTF confirmed or pending, BTC macro not strongly against
- WAIT: LTF not confirmed yet, OR PF marginal (1.0-1.3), OR mixed signals
- SKIP: blended PF < 1.0, OR recent verdict WEAKER, OR setup quality low, OR BEAR macro + weak edge

Be honest. If evidence is thin, recommend WAIT or SKIP — don't force a TAKE."""


# ============================================================================
# MODE TIMEFRAME LABELS  (mirror MODE_CONFIG from scanner.py)
# ============================================================================

_MODE_TF_LABELS: Dict[str, Dict[str, str]] = {
    "SWING": {"htf_label": "1D",  "ltf_label": "4H"},
    "DAY":   {"htf_label": "4H",  "ltf_label": "15m"},
    "SCALP": {"htf_label": "1H",  "ltf_label": "5m"},
}


# ============================================================================
# PUBLIC API — build_smc_prompt
# ============================================================================

def build_smc_prompt(
    symbol: str,
    mode: str,
    structure: Dict[str, Any],
    zones: Dict[str, Any],
    current_zone_classification: Dict[str, Any],
    ltf_confirmation: str,
    backtest: Dict[str, Any],
    btc_regime: str,
    fights_macro: bool,
    trade_plan: Dict[str, Any],
) -> str:
    """
    Build a structured prompt for the LLM.

    Args:
        symbol: e.g. "BTCUSDT"
        mode: "SWING" | "DAY" | "SCALP"
        structure: dict from classify_structure (has state, choch_bar, bos_bar, current_leg)
        zones: dict with smart_obs/fvgs/fibo/sr_levels lists
        current_zone_classification: dict from classify_current_price_in_zones
        ltf_confirmation: "CONFIRMED" | "PENDING" | "NONE"
        backtest: dict with per_coin/universe/blended/recent_check
        btc_regime: "BULL" | "CHOP" | "BEAR" | "UNKNOWN"
        fights_macro: bool — True if long setup vs BEAR macro
        trade_plan: dict with entry/sl/tp1/tp2/tp3/rr

    Returns:
        A string prompt (typically 1500-2500 chars) ready to send to LLM.
    """

    # ── Mode TF labels ────────────────────────────────────────────────────────
    tf_labels = _MODE_TF_LABELS.get(mode, {"htf_label": "?", "ltf_label": "?"})
    htf_label = tf_labels["htf_label"]
    ltf_label = tf_labels["ltf_label"]

    # ── fights_macro note ─────────────────────────────────────────────────────
    fights_macro_note = " ⚠ FIGHTS MACRO (long vs BEAR BTC)" if fights_macro else ""

    # ── Structure fields ──────────────────────────────────────────────────────
    state     = structure.get("state", "UNKNOWN")
    choch_bar = structure.get("choch_bar", "N/A")
    bos_bar   = structure.get("bos_bar", "N/A")
    current_leg = structure.get("current_leg") or {}
    leg_start_price = float(current_leg.get("leg_start_price", 0.0))
    leg_high_price  = float(current_leg.get("leg_high_price", 0.0))

    # ── Zone classification ───────────────────────────────────────────────────
    in_obs  = current_zone_classification.get("in_smart_ob", [])
    in_fvgs = current_zone_classification.get("in_fvg", [])
    in_fibo = current_zone_classification.get("in_fibo_786", False)
    in_srs  = current_zone_classification.get("at_sr_support", [])

    # Determine primary zone type label
    if in_obs:
        primary_zone_type = "Smart Order Block"
    elif in_fvgs:
        primary_zone_type = "Fair Value Gap"
    elif in_fibo:
        primary_zone_type = "Fibonacci 0.786"
    elif in_srs:
        primary_zone_type = "S/R Level"
    else:
        primary_zone_type = "None"

    # OB tier
    if in_obs:
        tiers = [ob.get("tier", "UNKNOWN") for ob in in_obs]
        ob_tier = "STRONG" if "STRONG" in tiers else tiers[0] if tiers else "N/A"
    else:
        ob_tier = "N/A"

    # FVG status
    if in_fvgs:
        statuses = [fvg.get("status", "UNKNOWN") for fvg in in_fvgs]
        fvg_status = statuses[0] if statuses else "N/A"
    else:
        fvg_status = "N/A"

    # SR touches
    sr_touches = in_srs[0].get("touches", "N/A") if in_srs else "N/A"

    # ── Current price (from trade_plan entry as proxy, or zones) ─────────────
    # We reconstruct from trade_plan since scanner doesn't pass current_price directly here.
    # Use entry_price as the current price reference (it's the zone level).
    entry_price = float(trade_plan.get("entry_price", 0.0))
    sl          = float(trade_plan.get("sl", 0.0))
    tp1_price   = float(trade_plan.get("tp1_price", 0.0))
    tp2_price   = float(trade_plan.get("tp2_price", 0.0))
    tp3_price   = float(trade_plan.get("tp3_price", 0.0))
    rr_to_tp2   = float(trade_plan.get("rr_to_tp2", 2.5))

    risk_pct = ((entry_price - sl) / entry_price * 100.0) if entry_price > 0 else 0.0

    # current_price: scanner stores it in trade_plan context; derive from entry
    current_price = entry_price  # best proxy available from trade_plan

    # ── Backtest fields ───────────────────────────────────────────────────────
    pc = backtest.get("per_coin", {})
    un = backtest.get("universe", {})
    bl = backtest.get("blended", {})
    rc = backtest.get("recent_check", {})

    wr_pc = float(pc.get("wr", 0.0))
    pf_pc = float(pc.get("pf", 0.0))
    mr_pc = float(pc.get("mean_r", 0.0))
    n_pc  = int(pc.get("n_setups", 0))

    wr_un = float(un.get("wr", 0.0))
    pf_un = float(un.get("pf", 0.0))
    mr_un = float(un.get("mean_r", 0.0))
    n_un  = int(un.get("n_setups", 0))

    wr_bl = float(bl.get("wr", 0.0))
    pf_bl = float(bl.get("pf", 0.0))
    mr_bl = float(bl.get("mean_r", 0.0))
    n_bl  = int(bl.get("n_effective", bl.get("n_setups", 0)))

    earlier_label  = rc.get("earlier_period_label", "N/A")
    recent_label   = rc.get("recent_period_label",  "N/A")
    recent_verdict = rc.get("verdict", "UNKNOWN")

    earlier_stats = rc.get("earlier_stats", {})
    recent_stats  = rc.get("recent_stats",  {})

    mr_e = float(earlier_stats.get("mean_r", 0.0))
    pf_e = float(earlier_stats.get("pf", 0.0))
    n_e  = int(earlier_stats.get("n", 0))

    mr_r = float(recent_stats.get("mean_r", 0.0))
    pf_r = float(recent_stats.get("pf", 0.0))
    n_r  = int(recent_stats.get("n", 0))

    # ── Render prompt ─────────────────────────────────────────────────────────
    return PROMPT_TEMPLATE.format(
        symbol=symbol,
        mode=mode,
        htf_label=htf_label,
        ltf_label=ltf_label,
        btc_regime=btc_regime,
        fights_macro_note=fights_macro_note,
        state=state,
        choch_bar=choch_bar,
        bos_bar=bos_bar,
        leg_start_price=leg_start_price,
        leg_high_price=leg_high_price,
        primary_zone_type=primary_zone_type,
        ob_tier=ob_tier,
        fvg_status=fvg_status,
        in_fibo_786=in_fibo,
        sr_touches=sr_touches,
        current_price=current_price,
        ltf_confirmation=ltf_confirmation,
        wr_pc=wr_pc, pf_pc=pf_pc, mr_pc=mr_pc, n_pc=n_pc,
        wr_un=wr_un, pf_un=pf_un, mr_un=mr_un, n_un=n_un,
        wr_bl=wr_bl, pf_bl=pf_bl, mr_bl=mr_bl, n_bl=n_bl,
        earlier_label=earlier_label, mr_e=mr_e, pf_e=pf_e, n_e=n_e,
        recent_label=recent_label,   mr_r=mr_r, pf_r=pf_r, n_r=n_r,
        recent_verdict=recent_verdict,
        entry=entry_price,
        sl=sl,
        risk_pct=risk_pct,
        tp1=tp1_price,
        tp2=tp2_price,
        tp3=tp3_price,
        rr=rr_to_tp2,
    )


# ============================================================================
# INTERNAL — API key resolution
# ============================================================================

def _resolve_api_key(provider: str, explicit: Optional[str]) -> Optional[str]:
    """
    Priority order:
    1. explicit arg if provided
    2. st.session_state["{provider}_api_key"]
    3. environment variable {PROVIDER}_API_KEY
    """
    if explicit:
        return explicit
    key_name = f"{provider}_api_key"
    if key_name in st.session_state and st.session_state[key_name]:
        return st.session_state[key_name]
    env_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(env_name)


# ============================================================================
# INTERNAL — LLM callers
# ============================================================================

def _call_groq(
    prompt: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    timeout: int = 20,
) -> str:
    """Returns raw response text from Groq."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(
    prompt: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: int = 20,
) -> str:
    """Returns raw response text from Anthropic."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ============================================================================
# INTERNAL — JSON parser (resilient)
# ============================================================================

def _parse_verdict_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tolerantly parse the verdict JSON from the LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - JSON embedded in surrounding text (find the first {...} block)
    - Missing fields (returns None if essential fields missing)
    """
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        content_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                content_lines.append(line)
        text = "\n".join(content_lines).strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Find the first { ... } block
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    # Validate essential fields
    if "verdict" not in obj or obj["verdict"] not in {"TAKE", "WAIT", "SKIP"}:
        return None
    if "reasoning" not in obj:
        return None

    # Fill defaults for optional fields
    obj.setdefault("confidence", 50)
    obj.setdefault("key_risks", [])

    return obj


# ============================================================================
# PUBLIC API — get_verdict
# ============================================================================

def get_verdict(
    prompt: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    """
    Send prompt to LLM and parse the verdict response.

    Args:
        prompt: output of build_smc_prompt()
        provider: "groq" | "anthropic" | "auto"
                  auto tries groq first, falls back to anthropic
        api_key: optional explicit key. If None, reads from:
                 - st.session_state.get("{provider}_api_key")
                 - env var {PROVIDER}_API_KEY as fallback
        model: optional model override
        timeout_seconds: HTTP timeout

    Returns:
        {
            "verdict":       "TAKE" | "WAIT" | "SKIP",
            "confidence":    int (0-100),
            "reasoning":     str,
            "key_risks":     List[str],
            "provider_used": str,
            "model_used":    str,
            "ai_error":      Optional[str],
        }
    """
    _defaults = {
        "groq":      "llama-3.3-70b-versatile",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    # ── Auto-fallback mode ────────────────────────────────────────────────────
    if provider == "auto":
        try:
            return get_verdict(prompt, "groq", api_key=None, model=model,
                               timeout_seconds=timeout_seconds)
        except Exception:
            pass
        try:
            return get_verdict(prompt, "anthropic", api_key=None, model=model,
                               timeout_seconds=timeout_seconds)
        except Exception as e:
            return {
                "verdict":       "WAIT",
                "confidence":    0,
                "reasoning":     "AI unavailable (both providers failed)",
                "key_risks":     [],
                "provider_used": "none",
                "model_used":    "none",
                "ai_error":      str(e),
            }

    # ── Single provider ───────────────────────────────────────────────────────
    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     f"AI unavailable (no {provider} API key)",
            "key_risks":     [],
            "provider_used": "none",
            "model_used":    "none",
            "ai_error":      f"No API key for {provider}",
        }

    effective_model = model or _defaults.get(provider, "unknown")

    try:
        if provider == "groq":
            response_text = _call_groq(prompt, resolved_key, effective_model, timeout_seconds)
        elif provider == "anthropic":
            response_text = _call_anthropic(prompt, resolved_key, effective_model, timeout_seconds)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    except Exception as e:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     f"AI call failed: {str(e)[:100]}",
            "key_risks":     [],
            "provider_used": provider,
            "model_used":    effective_model,
            "ai_error":      str(e),
        }

    parsed = _parse_verdict_json(response_text)
    if parsed is None:
        return {
            "verdict":       "WAIT",
            "confidence":    0,
            "reasoning":     "AI response unparseable",
            "key_risks":     [],
            "provider_used": provider,
            "model_used":    effective_model,
            "ai_error":      f"Parse failure: {response_text[:200]}",
        }

    parsed["provider_used"] = provider
    parsed["model_used"]    = effective_model
    parsed["ai_error"]      = None
    return parsed


# ============================================================================
# PUBLIC API — score_setup
# ============================================================================

def score_setup(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: take a scan_one_symbol() result dict, build prompt,
    get verdict, return a dict ready to be merged into the result.

    Args:
        result: output of scan_one_symbol()

    Returns:
        {
            "ai_verdict":    "TAKE" | "WAIT" | "SKIP",
            "ai_confidence": int,
            "ai_reasoning":  str,
            "ai_key_risks":  List[str],
            "ai_provider":   str,
            "ai_error":      Optional[str],
        }
    """
    _error_out = lambda msg: {
        "ai_verdict":    "WAIT",
        "ai_confidence": 0,
        "ai_reasoning":  msg,
        "ai_key_risks":  [],
        "ai_provider":   "none",
        "ai_error":      msg,
    }

    try:
        prompt = build_smc_prompt(
            symbol=result.get("symbol", "UNKNOWN"),
            mode=result.get("mode", "DAY"),
            structure=result.get("structure", {}),
            zones=result.get("zones", {}),
            current_zone_classification=result.get("current_zone_classification", {}),
            ltf_confirmation=result.get("ltf_confirmation", "NONE"),
            backtest=result.get("backtest", {}),
            btc_regime=result.get("btc_regime", "UNKNOWN"),
            fights_macro=result.get("fights_macro", False),
            trade_plan=result.get("trade_plan", {}),
        )
    except Exception as e:
        return _error_out(f"Prompt build failed: {e}")

    verdict_dict = get_verdict(prompt, provider="auto")

    return {
        "ai_verdict":    verdict_dict.get("verdict",    "WAIT"),
        "ai_confidence": verdict_dict.get("confidence", 0),
        "ai_reasoning":  verdict_dict.get("reasoning",  ""),
        "ai_key_risks":  verdict_dict.get("key_risks",  []),
        "ai_provider":   verdict_dict.get("provider_used", "none"),
        "ai_error":      verdict_dict.get("ai_error"),
    }
