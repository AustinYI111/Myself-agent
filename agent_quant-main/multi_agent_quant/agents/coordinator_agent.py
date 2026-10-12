from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
import math


@dataclass
class CoordinatorAgent:
    # ====== 静态先验权重（主策略的“初始信任”）======
    agent_weights: Dict[str, float]

    # ====== 融合决策阈值 ======
    min_edge: float = 0.05
    min_score_to_trade: float = 0.05
    regime_boost: float = 0.25
    min_conf_when_trade: float = 0.20  # 融合输出最低可交易置信度（兜底）

    # ====== 动态权重：表现跟踪参数（只对主策略生效） ======
    perf_alpha: float = 0.10
    perf_clip: float = 0.03
    perf_temperature: float = 0.50
    dyn_blend: float = 0.60
    min_weight_floor: float = 0.02

    # ✅ 辅助Agent（默认 ml 是辅助，不进入权重池）
    aux_agents: Set[str] = field(default_factory=lambda: {"ml"})

    # ✅ 路线A：ML 否决器开关与 margin
    ml_veto_enabled: bool = True
    ml_veto_margin: float = 0.03  # 建议 0.02~0.05

    # 内部状态：每个主 agent 的“近期表现分数”（EWMA）
    perf_ewma: Dict[str, float] = field(default_factory=dict)

    # --------------------------
    # 基础工具
    # --------------------------
    def _normalize(self, w: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(v, 0.0) for v in w.values())
        if s <= 0:
            n = len(w) if len(w) else 1
            return {k: 1.0 / n for k in w.keys()}
        return {k: max(v, 0.0) / s for k, v in w.items()}

    def _main_weight_pool(self) -> Dict[str, float]:
        """主权重池：排除 aux_agents（比如 ml），避免稀释票权。"""
        return {k: v for k, v in self.agent_weights.items() if k not in self.aux_agents}

    def _apply_regime(self, weights: Dict[str, float], market_state: Optional[str]) -> Dict[str, float]:
        """
        market_state: 'trend' | 'range' | 'high_vol' | None
        这里只对主权重池做偏置。
        """
        w = dict(weights)

        if market_state == "trend":
            if "trend" in w:
                w["trend"] = w["trend"] + self.regime_boost
            if "mean_reversion" in w:
                w["mean_reversion"] = max(0.0, w["mean_reversion"] - self.regime_boost)

        elif market_state == "range":
            if "mean_reversion" in w:
                w["mean_reversion"] = w["mean_reversion"] + self.regime_boost
            if "trend" in w:
                w["trend"] = max(0.0, w["trend"] - self.regime_boost)

        return self._normalize(w)

    # --------------------------
    # 动态权重（表现驱动）——只对主策略生效
    # --------------------------
    def _ensure_perf_keys(self) -> None:
        for k in self._main_weight_pool().keys():
            self.perf_ewma.setdefault(k, 0.0)

    def update_performance(self, agent_outputs: Dict[str, Dict[str, Any]], next_ret: float) -> None:
        """
        next_ret：下一根 bar 的收益率 (close[t+1]/close[t]-1)
        给每个“主策略”一个即时 reward：
          - buy  -> + next_ret
          - sell -> - next_ret
          - hold -> 0
        再做 EWMA 更新。
        """
        self._ensure_perf_keys()
        r = max(-self.perf_clip, min(self.perf_clip, float(next_ret)))

        for name, out in agent_outputs.items():
            # ✅ 辅助 agent 不进入表现权重竞争（避免噪声）
            if name in self.aux_agents:
                continue

            sig = str(out.get("signal", "hold")).lower()
            conf = float(out.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))

            if sig == "buy":
                reward = +r * conf
            elif sig == "sell":
                reward = -r * conf
            else:
                reward = 0.0

            old = self.perf_ewma.get(name, 0.0)
            new = (1.0 - self.perf_alpha) * old + self.perf_alpha * reward
            self.perf_ewma[name] = float(new)

    def _softmax_weights(self) -> Dict[str, float]:
        """
        把 perf_ewma -> 动态权重（softmax），只对主策略。
        """
        self._ensure_perf_keys()

        tau = max(1e-6, float(self.perf_temperature))
        keys = list(self._main_weight_pool().keys())
        vals = {k: self.perf_ewma.get(k, 0.0) / tau for k in keys}

        m = max(vals.values()) if vals else 0.0
        exps = {k: math.exp(v - m) for k, v in vals.items()}
        s = sum(exps.values()) if exps else 0.0
        if s <= 0:
            return self._normalize(dict(self._main_weight_pool()))

        w = {k: exps[k] / s for k in exps.keys()}
        w = {k: max(self.min_weight_floor, w[k]) for k in w.keys()}
        return self._normalize(w)

    def _mix_static_dynamic(self, base: Dict[str, float]) -> Dict[str, float]:
        """
        base：主策略静态+regime后的权重
        dyn ：主策略表现驱动权重
        """
        dyn = self._softmax_weights()
        b = max(0.0, min(1.0, float(self.dyn_blend)))

        mixed = {}
        keys = set(base.keys()) | set(dyn.keys())
        for k in keys:
            mixed[k] = (1.0 - b) * float(base.get(k, 0.0)) + b * float(dyn.get(k, 0.0))
        return self._normalize(mixed)

    # --------------------------
    # 辅助（ML）权重：不进入权重池，只用于展示/否决器
    # --------------------------
    def _aux_weight(self, name: str, market_state: Optional[str]) -> float:
        """
        ✅ BUGFIX:
        - 如果 ml 根本不在 agent_weights（或权重<=0），必须返回 0，
          不能因为 regime_boost “凭空加出” 0.125 这种值。
        """
        if name not in self.agent_weights:
            return 0.0

        base = float(self.agent_weights.get(name, 0.0))
        if base <= 0.0:
            return 0.0

        w = base
        if market_state == "trend":
            w = w + self.regime_boost * 0.5
        elif market_state == "range":
            w = max(0.0, w - self.regime_boost * 0.25)

        return float(w)

    # --------------------------
    # 融合决策（路线A：ML 否决器）
    # --------------------------
    def aggregate(self, agent_outputs: Dict[str, Dict[str, Any]], market_state: Optional[str] = None) -> Dict[str, Any]:
        """
        主策略先融合得到 best_sig；
        ML 仅作为否决器（veto）：把低胜率/方向冲突的信号打回 hold。
        """
        # 1) 主策略静态权重池（排除 ml）
        base_main = self._normalize(self._main_weight_pool())

        # 2) 主策略 regime 偏置
        w_regime_main = self._apply_regime(base_main, market_state)

        # 3) 主策略混入动态权重
        w_main = self._mix_static_dynamic(w_regime_main)

        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        details: Dict[str, Any] = {}

        # ML 信息（仅用于 veto，不进 score）
        ml_p_up: Optional[float] = None
        ml_aux_meta: Optional[Dict[str, Any]] = None
        ml_conf: float = 0.0

        # 4) 主策略投票（ml 不投票）
        for name, out in agent_outputs.items():
            meta = out.get("meta", {}) or {}
            meta = dict(meta) if isinstance(meta, dict) else {}

            if name in self.aux_agents and meta.get("aux", False):
                ml_aux_meta = meta
                ml_conf = float(out.get("confidence", 0.0))
                # p_up 可能 None（insufficient_data 时）
                try:
                    ml_p_up = meta.get("p_up", None)
                    if ml_p_up is not None:
                        ml_p_up = float(ml_p_up)
                except Exception:
                    ml_p_up = None

                details[name] = {
                    "signal": "aux",
                    "confidence": max(0.0, min(1.0, ml_conf)),
                    "weight": self._aux_weight(name, market_state),
                    "contrib": 0.0,  # ✅ 路线A：不做 score 微调
                    "aux_suggest": meta.get("aux_suggest", "hold"),
                    "aux_strength": meta.get("aux_strength", 0.0),
                    "p_up": ml_p_up,
                    "reason": meta.get("reason", None),
                }
                continue

            sig = str(out.get("signal", "hold")).lower()
            if sig not in score:
                sig = "hold"

            conf = float(out.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            weight = float(w_main.get(name, 0.0))

            contrib = weight * conf
            score[sig] += contrib
            details[name] = {"signal": sig, "confidence": conf, "weight": weight, "contrib": contrib}

        # 5) 先按主策略选 best
        best_sig = max(score, key=score.get)
        best = float(score[best_sig])
        second = sorted(score.values(), reverse=True)[1]
        edge = float(best - second)

        if best <= 0.0:
            best_sig = "hold"
            edge = 0.0

        min_trade = float(self.min_score_to_trade)
        if market_state == "high_vol":
            min_trade += 0.10

        if best_sig != "hold":
            if best < min_trade or edge < float(self.min_edge):
                best_sig = "hold"

        # 6) ✅ 路线A：ML 否决器（veto）
        veto_applied = False
        veto_reason = None
        margin = max(0.0, float(self.ml_veto_margin))

        if self.ml_veto_enabled and best_sig in ("buy", "sell") and (ml_p_up is not None):
            # buy：p_up < 0.5 + margin 否决
            if best_sig == "buy" and ml_p_up < (0.5 + margin):
                best_sig = "hold"
                veto_applied = True
                veto_reason = f"veto_buy(p_up={ml_p_up:.4f} < {0.5+margin:.4f})"

            # sell：p_up > 0.5 - margin 否决
            elif best_sig == "sell" and ml_p_up > (0.5 - margin):
                best_sig = "hold"
                veto_applied = True
                veto_reason = f"veto_sell(p_up={ml_p_up:.4f} > {0.5-margin:.4f})"

        # 7) 输出置信度
        if best_sig == "hold":
            out_conf = 0.05
        else:
            supporters = [
                float(v.get("confidence", 0.0))
                for v in details.values()
                if v.get("signal") == best_sig
            ]
            out_conf = max(supporters) if supporters else best
            out_conf = max(float(self.min_conf_when_trade), min(1.0, float(out_conf)))

        return {
            "signal": best_sig,
            "confidence": float(out_conf),
            "meta": {
                "market_state": market_state,
                "score": score,
                "edge": edge,
                "weights": w_main,  # ✅ 只包含主策略
                "ml_aux_weight": self._aux_weight("ml", market_state),  # ✅ no-ML 时会是 0.0
                "ml_veto": {
                    "enabled": bool(self.ml_veto_enabled),
                    "margin": float(margin),
                    "applied": bool(veto_applied),
                    "reason": veto_reason,
                    "p_up": ml_p_up,
                },
                "perf_ewma": dict(self.perf_ewma),  # ✅ 只会包含主策略
                "details": details,
            },
        }