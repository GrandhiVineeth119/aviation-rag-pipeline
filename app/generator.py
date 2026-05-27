"""
generator.py — Calls the LLM with a strict grounding system prompt.
 
Providers:
  • ollama    — local, no API key (default)
  • openai    — cloud, requires OPENAI_API_KEY
  • anthropic — cloud, requires ANTHROPIC_API_KEY
 
Ollama optimisations:
  • Single LLM call only (self_verify disabled for local models)
  • Chunk text capped at 600 chars
  • Hard cap on total prompt size (8000 chars)
  • num_ctx=4096, num_predict=512
"""
 
import re, json, urllib.request, urllib.error
from app.config import GeneratorConfig, REFUSAL_MESSAGE
from app.document_processor import DocumentChunk
 
SYSTEM_PROMPT = """\
You are an Aviation Document AI assistant.
Answer ONLY from the CONTEXT below. Follow every rule exactly.
 
RULES:
1. Use ONLY information written in the CONTEXT. Never use outside knowledge.
2. Cite every fact with the chunk id like this: [chunk_id]
3. If context is insufficient, reply exactly: INSUFFICIENT_CONTEXT
4. Non-aviation questions (art, geography, sports, trivia): reply INSUFFICIENT_CONTEXT
5. Do not guess or speculate. Be concise.
 
CONTEXT:
{context}
 
CHUNK IDS:
{chunk_ref}
"""
 
_MAX_CHARS   = 8_000   
_CHUNK_CHARS = 600     
 
 
class Generator:
    def __init__(self, config: GeneratorConfig | None = None):
        self.cfg = config or GeneratorConfig()
        self._client = None
 
 
 
    def generate(self, query: str, retrieved: list[dict]) -> dict:
        if not retrieved:
            return _refuse("no_retrieved_chunks")
 
        ctx_parts, ref_parts = [], []
        for r in retrieved:
            c: DocumentChunk = r["chunk"]
            ctx_parts.append(f"[{c.chunk_id}] ({c.citation()})\n{c.text[:_CHUNK_CHARS]}")
            ref_parts.append(f"{c.chunk_id} -> {c.citation()}")
 
        system = SYSTEM_PROMPT.format(
            context="\n\n".join(ctx_parts),
            chunk_ref="\n".join(ref_parts),
        )
 
        if len(system) + len(query) > _MAX_CHARS:
            system = system[:(_MAX_CHARS - len(query) - 50)] + "\n[truncated]"
 
        answer = self._call(system, query)
 
        if "INSUFFICIENT_CONTEXT" in answer:
            return _refuse("llm_insufficient_context")
 
        cited_ids = set(re.findall(r"\[([a-f0-9]{12})\]", answer))
        id_map = {r["chunk"].chunk_id: r["chunk"] for r in retrieved}
        citations = [
            {
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "snippet": c.text[:200],
            }
            for cid in cited_ids
            if (c := id_map.get(cid))
        ]
        return {"answer": answer, "citations": citations, "refused": False}
 
    def self_verify(self, query: str, answer: str, retrieved: list[dict]) -> dict:
        """
        For Ollama: skip the second LLM call — it doubles latency on local hardware.
        For cloud providers: do the full verification.
        """
        if self.cfg.provider == "ollama":
            return {"supported": True}  
 
        ctx = "\n\n".join(
            f"[{r['chunk'].chunk_id}]: {r['chunk'].text[:300]}" for r in retrieved
        )
        prompt = (
            f"CONTEXT:\n{ctx}\n\nQUESTION: {query}\nANSWER: {answer}\n\n"
            "Is EVERY claim in the ANSWER directly supported by the CONTEXT?\n"
            "Reply ONLY: SUPPORTED or NOT_SUPPORTED"
        )
        resp = self._call("You are a strict aviation fact-checker.", prompt)
        return {"supported": "NOT_SUPPORTED" not in resp.upper()}
 
    
    def _call(self, system: str, user: str) -> str:
        if self.cfg.provider == "ollama":
            return self._call_ollama(system, user)
        elif self.cfg.provider == "openai":
            return self._call_openai(system, user)
        elif self.cfg.provider == "anthropic":
            return self._call_anthropic(system, user)
        raise ValueError(f"Unknown provider: {self.cfg.provider!r}")
 
   
 
    def _call_ollama(self, system: str, user: str) -> str:
        url = f"{self.cfg.ollama_base_url}/api/chat"
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 512,
                "num_ctx": 4096,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama HTTP {e.code}: {body_text}\n"
                f"Model: {self.cfg.model} — run 'ollama list' to confirm."
            ) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Ollama request failed: {e}\n"
                f"• Is Ollama running?  Check: http://localhost:11434\n"
                f"• Is the model ready? Run: ollama list"
            ) from e
 
    
    @property
    def _openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.cfg.api_key)
        return self._client
 
    def _call_openai(self, system: str, user: str) -> str:
        r = self._openai_client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return r.choices[0].message.content.strip()
 
 
    @property
    def _anthropic_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.cfg.api_key)
        return self._client
 
    def _call_anthropic(self, system: str, user: str) -> str:
        r = self._anthropic_client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text.strip()
 
 
def _refuse(reason: str) -> dict:
    return {"answer": REFUSAL_MESSAGE, "citations": [], "refused": True, "_reason": reason}