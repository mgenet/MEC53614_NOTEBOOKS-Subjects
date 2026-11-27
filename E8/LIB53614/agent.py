#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet                                                  ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""
A minimal coding agent, and its harness (E8, draft).

The agent is a loop around a language model, served by any OpenAI-compatible
endpoint (e.g., a local model served by vLLM, llama.cpp or Ollama):
    1. the context window is assembled: system prompt + harness + task (+ the
       previous attempts and the feedback they received);
    2. the model answers with a python script (in a ```python block);
    3. the script is saved and executed with the requested command line;
    4. the hooks (deterministic python functions) check the result;
    5. if the script crashes or a hook fails, the error messages are sent back
       to the model, and the loop starts again (at most `max_turns` times).

The harness gathers the components of the "harness engineering" workshop:
    - agents_md: the identity card of the project, always in the context;
    - rules: short constraints, always in the context;
    - skills: complete procedures (here always loaded; in real tools, loaded
      when their description matches the request);
    - hooks: deterministic checks, run by the loop (not by the model).
Commands and sub-agents are not implemented here.

Everything that is sent to the model is in `Run.messages`: the model only sees
the context window.
"""

import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
import time

SYSTEM_PROMPT = """You are a coding assistant. Answer with exactly one complete python script, in a single ```python code block, and nothing else after it."""

@dataclasses.dataclass
class Harness:
    agents_md: str = ""
    rules: list = dataclasses.field(default_factory=list)   # list of strings
    skills: dict = dataclasses.field(default_factory=dict)  # name -> text
    hooks: list = dataclasses.field(default_factory=list)   # functions hook(script_path, output) -> (ok, message)

    @classmethod
    def from_folder(cls, folder, hooks=()):
        """Reads AGENTS.md, rules/*.md and skills/*/SKILL.md from a folder (as in real tools)."""
        folder = pathlib.Path(folder)
        read = lambda p: p.read_text(encoding="utf-8") if p.exists() else ""
        return cls(agents_md=read(folder / "AGENTS.md"),
                   rules=[read(p) for p in sorted((folder / "rules").glob("*.md"))],
                   skills={p.parent.name: read(p) for p in sorted((folder / "skills").glob("*/SKILL.md"))},
                   hooks=list(hooks))

    def context(self):
        parts = []
        if self.agents_md: parts.append("# Project (AGENTS.md)\n\n" + self.agents_md)
        if self.rules:     parts.append("# Rules\n\n" + "\n\n".join(self.rules))
        if self.skills:    parts.append("# Skills\n\n" + "\n\n".join(f"## {name}\n\n{text}" for name, text in self.skills.items()))
        return "\n\n".join(parts)

@dataclasses.dataclass
class Run:
    messages: list
    scripts: list
    outputs: list
    feedback: list
    success: bool = False
    n_calls: int = 0
    n_tokens: int = 0
    duration: float = 0.

    @property
    def result(self):
        """Last JSON output of the script (dict), or None."""
        for out in reversed(self.outputs):
            if isinstance(out, dict): return out
        return None

def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else text

def execute(script_path, args=(), timeout=300):
    """Runs a script; returns its last line of output parsed as JSON (dict), or an error message (str)."""
    try:
        proc = subprocess.run([sys.executable, str(script_path), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout} s."
    if (proc.returncode != 0):
        return "The script failed:\n" + proc.stderr[-3000:]
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    try:
        out = json.loads(lines[-1])
        assert isinstance(out, dict)
        return out
    except Exception:
        return "The last line of the output is not a JSON dictionary:\n" + proc.stdout[-1000:]

def run(task, harness=None, client=None, model=None, args=(), max_turns=5, workdir="agent_runs", name="run", temperature=0.7, timeout=300, verbose=True):
    """Runs the agent on the task, with the harness (or without: harness=None)."""
    harness = harness or Harness()
    client = client or default_client()
    model = model or os.environ.get("MEC53614_LLM_MODEL", "default")
    workdir = pathlib.Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    system = SYSTEM_PROMPT + ("\n\n" + harness.context() if harness.context() else "")
    r = Run(messages=[{"role": "system", "content": system}, {"role": "user", "content": task}], scripts=[], outputs=[], feedback=[])
    t0 = time.time()
    for turn in range(max_turns):
        response = client.chat.completions.create(model=model, messages=r.messages, temperature=temperature)
        r.n_calls += 1
        r.n_tokens += getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        answer = response.choices[0].message.content
        r.messages.append({"role": "assistant", "content": answer})
        path = workdir / f"{name}_turn{turn}.py"
        path.write_text(extract_code(answer), encoding="utf-8")
        r.scripts.append(path)
        out = execute(path, args, timeout)
        r.outputs.append(out)
        if isinstance(out, str):
            messages = [out]
        else:
            messages = [msg for (ok, msg) in (hook(path, out) for hook in harness.hooks) if not ok]
        r.feedback.append(messages)
        if verbose: print(f"[{name}] turn {turn}: " + ("OK" if not messages else f"{len(messages)} problem(s)") + (f", K_t = {out.get('K_t')}" if isinstance(out, dict) else ""))
        if not messages:
            r.success = True
            break
        r.messages.append({"role": "user", "content": "Your script has the following problems; fix them and answer with the complete corrected script.\n\n" + "\n\n".join(f"- {m}" for m in messages)})
    r.duration = time.time() - t0
    return r

def default_client():
    """OpenAI-compatible client, configured by the environment variables
    MEC53614_LLM_URL (e.g., http://llm.example:8000/v1) and MEC53614_LLM_KEY."""
    import openai
    return openai.OpenAI(base_url=os.environ.get("MEC53614_LLM_URL", "http://localhost:8000/v1"), api_key=os.environ.get("MEC53614_LLM_KEY", "none"))

class ScriptedClient:
    """A fake client that answers with predefined texts, in order (for testing
    the loop and the hooks without a language model)."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.chat = self
        self.completions = self
    def create(self, model=None, messages=None, **kwargs):
        text = self.answers.pop(0) if (len(self.answers) > 1) else self.answers[0]
        message = type("Message", (), {"content": text})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": None})()

class SimulatedClient:
    """Offline stand-in for a language model, for testing the loop and the
    hooks when no model is available: it answers with one of the scripts of
    `LIB53614.plate` (correct or faulty, at random); after a feedback message,
    it answers with the correct script with probability p_fix. It ignores the
    harness: it is not a model!"""
    def __init__(self, seed=0, p_fix=0.7):
        import numpy
        from . import plate
        self.rng = numpy.random.default_rng(seed)
        self.p_fix = p_fix
        self.variants = list(plate.VARIANTS)
        self.script = plate.script
        self.chat = self
        self.completions = self
    def create(self, model=None, messages=None, **kwargs):
        feedback = (messages is not None) and (messages[-1]["role"] == "user") and messages[-1]["content"].startswith("Your script has")
        variant = "reference" if (feedback and (self.rng.random() < self.p_fix)) else str(self.rng.choice(self.variants))
        text = "Here is the script.\n\n```python\n" + self.script(variant) + "```"
        message = type("Message", (), {"content": text})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": None})()
