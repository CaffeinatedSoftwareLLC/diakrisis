"""espresso-380m: typed decisions on CPU with ONNX Runtime. No PyTorch needed.

    from espresso import Espresso
    m = Espresso("path/to/espresso-380m")                 # fp32 (exact); Espresso(path, precision="w8") for the 8-bit build
    m.decide("I was charged twice this month.", {
        "billing": {"type": "noul", "instructions": "Is this about billing?"},
        "team": {"type": "choice", "instructions": "Which team handles it?",
                 "criteria": {"billing": "Payments and refunds", "tech": "Bugs and outages"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
    })

Each answer has the model's probabilities and a `decision`. On choice questions that list "none_of_these" and have at least
50 options, the frozen decision rule applies: a small boost to none_of_these, and decision "REVIEW" (send to a person)
when the top answer's probability is below 0.88. A choice over exactly {PASS, FAIL} gets a small shift against PASS.

The input rendering (ub_d3_common.to_row), the decision heads and the rule are line-for-line ports of the code the model
was trained and evaluated with (vela3-fast DecisionEncoder; r7_calibrate.apply_decision_biases; r8_decide).
"""
import json
import math
from pathlib import Path

import numpy as np

REVIEW = "REVIEW"
NONE_KEY, PASS, FAIL = "none_of_these", "PASS", "FAIL"
LOG_FLOOR = 1e-12
NOUL_OPTIONS = {"no": "", "yes": ""}


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _opt_text(name, desc):
    return name if not desc else f"{name}: {desc}"


def _options(q):
    """Options in criteria order, as the model reads them. score -> one option per level (the level text)."""
    t = q.get("type")
    if t == "noul":
        return dict(NOUL_OPTIONS)
    c = q.get("criteria")
    if t == "choice":
        if isinstance(c, dict):
            return {str(k): ("" if v is None else _text(v)) for k, v in c.items()}
        return {str(k): "" for k in c}
    if t == "score":
        if isinstance(c, dict):
            return {str(k): ("" if v is None else _text(v)) for k, v in c.items()}
        return {str(k): "" for k in c}
    raise ValueError(f"unsupported question type {t!r} (noul, choice, score)")


def apply_rule(probs, rule):
    """-> (probabilities after the rule, decision | REVIEW). probs: {option: p} in the question's option order."""
    opts = list(probs)
    scoped = NONE_KEY in opts and len(opts) >= rule.get("min_options", 0)
    d = rule["delta_none"] if (scoped and NONE_KEY in opts) else 0.0
    bb = rule["b_pass"] if (len(opts) == 2 and set(opts) == {PASS, FAIL}) else 0.0
    if d == 0 and bb == 0:
        new = dict(probs)
        best = max(opts, key=lambda k: (probs[k], -opts.index(k)))  # first maximum
    else:
        logs = {k: math.log(max(probs[k], LOG_FLOOR)) for k in opts}
        if NONE_KEY in logs:
            logs[NONE_KEY] += d
        if bb:
            logs[PASS] += bb
        top = max(logs.values())
        z = top + math.log(sum(math.exp(v - top) for v in logs.values()))
        new = {k: math.exp(v - z) for k, v in logs.items()}
        best = opts[0]
        for k in opts[1:]:
            if new[k] > new[best]:
                best = k
    if rule["tau"] and scoped and new[best] / sum(new.values()) < rule["tau"]:
        return new, REVIEW
    return new, best


class Espresso:
    def __init__(self, path, precision="fp32", threads=0):
        import onnxruntime as ort
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        p = Path(path)
        self.cfg = json.loads((p / "espresso_config.json").read_text(encoding="utf-8"))
        self.rule = json.loads((p / "decision_rule.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(p / "tokenizer.json"))
        self.mk = self.cfg["marker_ids"]
        self.bos, self.eos, self.pad = self.cfg["bos_token_id"], self.cfg["eos_token_id"], self.cfg["pad_token_id"]
        self.max_len, self.budget = self.cfg["max_len"], self.cfg["token_budget"]
        h = load_file(str(p / "heads.safetensors"))
        self.h = {k: v.astype(np.float32) for k, v in h.items()}
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        f = {"fp32": "encoder_fp32.onnx", "w8": "encoder_w8.onnx"}[precision]
        self.sess = ort.InferenceSession(str(p / f), so, providers=["CPUExecutionProvider"])

    def _ids(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids if s else []

    def _record(self, state_ids, q):
        opts = _options(q)
        ids = [self.bos, self.mk["[Q]"]] + self._ids(_text(q.get("instructions", "")))
        opt_pos = []
        for n, d in opts.items():
            opt_pos.append(len(ids))
            ids += [self.mk["[O]"]] + self._ids(_opt_text(n, d))
        ids.append(self.mk["[SEP_SCHEMA]"])
        ps = len(ids)
        ids += [self.mk["[SEG_user]"]] + state_ids
        pe = len(ids)
        ids.append(self.eos)
        if len(ids) > self.max_len:
            raise ValueError(f"input is {len(ids)} tokens; the limit is {self.max_len}")
        return {"ids": ids, "opt_pos": opt_pos, "names": list(opts), "pool": (ps, pe)}

    def _heads(self, H, r):
        """One record's option logits from its hidden states H [L, 1024] (DecisionEncoder._logits)."""
        h = self.h

        def ln(x):
            m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
            return (x - m) / np.sqrt(v + 1e-5) * h["norm.weight"] + h["norm.bias"]

        def lin(x, k):
            return x @ h[f"{k}.weight"].T + h[f"{k}.bias"]

        def unit(x):
            return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

        ps, pe = r["pool"]
        pooled = H[ps:pe].astype(np.float64).sum(0) / max(pe - ps, 1)
        v = unit(lin(ln(pooled.astype(np.float32)), "w_p"))
        uq = lin(ln(H[1]), "w_q")
        O = H[r["opt_pos"]]
        u = unit(lin(ln(O), "w_o") + uq)
        mlp = np.maximum(lin(O, "cls_mlp.0"), 0) @ h["cls_mlp.3.weight"].T + h["cls_mlp.3.bias"]
        return (u @ v) / math.exp(float(h["log_tau"])) + mlp[:, 0]

    def _forward(self, recs):
        out, order, i = [None] * len(recs), sorted(range(len(recs)), key=lambda k: len(recs[k]["ids"])), 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(recs[order[j]]["ids"]) * (j - i + 1) <= self.budget:
                j += 1
            grp = [recs[k] for k in order[i:j]]
            w = max(len(r["ids"]) for r in grp)
            x = np.full((len(grp), w), self.pad, dtype=np.int64)
            msk = np.zeros_like(x)
            for b, r in enumerate(grp):
                x[b, :len(r["ids"])] = r["ids"]
                msk[b, :len(r["ids"])] = 1
            H = self.sess.run(None, {"input_ids": x, "attention_mask": msk})[0]
            for b, k in enumerate(order[i:j]):
                z = self._heads(H[b], recs[k]).astype(np.float64)
                p = np.exp(z - z.max())
                out[k] = p / p.sum()
            i = j
        return out

    def probabilities(self, state, questions):
        """{key: {option: p}} straight from the model (no rule). noul options are {"no", "yes"}."""
        st = self._ids(_text(state if state is not None else ""))
        keys = list(questions)
        recs = [self._record(st, questions[k]) for k in keys]
        return {k: {n: float(v) for n, v in zip(r["names"], p)} for k, r, p in zip(keys, recs, self._forward(recs))}

    def decide(self, state, questions):
        """{key: answer}. noul: {"noul": p(yes), "decision": "yes"|"no"}; choice: {"probabilities", "decision"} (decision may be
        "REVIEW"); score: {"probabilities" by level, "score": expected level index, "decision": top level}."""
        out = {}
        for k, probs in self.probabilities(state, questions).items():
            t = questions[k]["type"]
            if t == "noul":
                out[k] = {"type": "noul", "noul": probs["yes"], "decision": "yes" if probs["yes"] > probs["no"] else "no"}
            elif t == "choice":
                new, dec = apply_rule(probs, self.rule)
                out[k] = {"type": "choice", "probabilities": new, "decision": dec,
                          "confidence": None if dec == REVIEW else new[dec]}
            else:
                levels = list(probs)
                out[k] = {"type": "score", "probabilities": probs, "score": sum(i * probs[n] for i, n in enumerate(levels)),
                          "decision": max(levels, key=lambda n: (probs[n], -levels.index(n)))}
        return out
