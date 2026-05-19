import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Union, Literal

# Optional imports for lightweight alternatives
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


RewardMethod = Literal["llm", "rouge", "cosine", "combined"]


class HuggingFaceLLMReward:
    """
    Reward function with multiple scoring backends:
      - "llm"      : OLMo / Qwen generation-based scoring (original)
      - "rouge"    : ROUGE-L F1 score (requires `pip install rouge-score`)
      - "cosine"   : TF-IDF cosine similarity (requires `pip install scikit-learn`)
      - "combined" : Average of ROUGE-L and cosine similarity
    """

    def __init__(
        self,
        method: RewardMethod = "rouge",
        model_name: str = "Qwen/Qwen2.5-7B",
        device: str = "cuda",
        max_length: int = 512,
    ):
        """
        Parameters
        ----------
        method : one of "llm" | "rouge" | "cosine" | "combined"
            Scoring backend to use.  LLM is the original behaviour; the
            others are fast, dependency-light alternatives that require no
            GPU and no model download.

        model_name : HuggingFace model ID (only used when method == "llm")
            OLMo options  : "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf"
            Qwen options  : "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B",
                            "Qwen/Qwen2-7B", "Qwen/Qwen1.5-0.5B-Chat"
            Recommended   : "Qwen/Qwen2-0.5B" for speed,
                            "allenai/OLMo-1B-hf" for balance.

        device     : "cuda" or "cpu"  (only used when method == "llm")
        max_length : max tokeniser length  (only used when method == "llm")
        """
        self.method = method

        if method == "llm":
            self._init_llm(model_name, device, max_length)
        elif method == "rouge":
            self._init_rouge()
        elif method == "cosine":
            self._init_cosine()
        elif method == "combined":
            self._init_rouge()
            self._init_cosine()
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: llm, rouge, cosine, combined")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_llm(self, model_name: str, device: str, max_length: int):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"Loading lightweight LLM reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        print(f"Model loaded on {self.device}")

    def _init_rouge(self):
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score is required for method='rouge'. Run: pip install rouge-score")
        # rougeL balances precision and recall without requiring contiguous n-grams
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        print("ROUGE-L scorer ready (no model download needed)")

    def _init_cosine(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for method='cosine'. Run: pip install scikit-learn")
        print("TF-IDF cosine scorer ready (no model download needed)")

    # ------------------------------------------------------------------
    # Public interface  – same signature as the original class
    # ------------------------------------------------------------------

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
    ) -> torch.Tensor:
        """Compute per-sample rewards and return a float32 tensor of shape (N,)."""
        if self.method == "llm":
            return self._score_llm(responses, ground_truths)
        elif self.method == "rouge":
            return self._score_rouge(responses, ground_truths)
        elif self.method == "cosine":
            return self._score_cosine(responses, ground_truths)
        elif self.method == "combined":
            return self._score_combined(responses, ground_truths)

    # ------------------------------------------------------------------
    # Scoring backends
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_llm(self, responses: List[str], ground_truths: List[str]) -> torch.Tensor:
        """Original generation-based LLM scoring."""
        rewards = []
        for response, gt in zip(responses, ground_truths):
            print(f"Computing ROUGE for:\nGT: {gt}\nRESP: {response}")
            prompt = (
                "Rate how similar in meaning the Response is to the Reference. "
                "Ignore length differences. Output ONLY a numerical score from 0.0 to 1.0\n\n"
                "- 0.0-0.3: Poor (mismatch)\n"
                "- 0.4-0.6: Moderate (captures a few key details)\n"
                "- 0.7-0.9: Good (captures most key details)\n"
                "- 1.0: Perfect (matches meaning perfectly)\n\n"
                f"Reference: {gt}\n"
                f"Response: {response[0]}\n\n"
                "Score:"
            )
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                score = self._extract_score(generated)
                print(f"Generated score: {generated}")
                rewards.append(score)
            except Exception as e:
                print(f"LLM reward error: {e}")
                rewards.append(0.5)

        return torch.tensor(rewards, dtype=torch.float32)

    def _score_rouge(self, responses: List[str], ground_truths: List[str]) -> torch.Tensor:
        """
        ROUGE-L F1 score.

        ROUGE-L measures longest common subsequence overlap between the
        prediction and reference, naturally handling paraphrase and word-
        order variation better than exact n-gram matches (ROUGE-1/2).
        Returns values in [0, 1].
        """
        rewards = []
        for response, gt in zip(responses, ground_truths):
            try:
                print(f"Computing ROUGE for:\nGT: {gt}\nRESP: {response[0]}")
                scores = self._rouge.score(gt, response[0])
                # fmeasure is the harmonic mean of precision and recall
                rewards.append(scores["rougeL"].fmeasure)
            except Exception as e:
                print(f"ROUGE reward error: {e}")
                rewards.append(0.5)
        return torch.tensor(rewards, dtype=torch.float32)

    def _score_cosine(self, responses: List[str], ground_truths: List[str]) -> torch.Tensor:
        """
        TF-IDF cosine similarity.

        Fits a TF-IDF vocabulary over the current batch (reference + prediction
        pairs) then measures the cosine angle between each pair.  This captures
        vocabulary overlap while down-weighting common stopwords.
        Returns values in [0, 1].
        """
        rewards = []
        for response, gt in zip(responses, ground_truths):
            try:
                # Need at least two strings to fit the vectoriser
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([gt, response])
                score = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
                # Clamp to [0, 1] – cosine on non-negative TF-IDF is already ≥ 0
                score = max(0.0, min(1.0, score))
                rewards.append(score)
            except Exception as e:
                print(f"Cosine reward error: {e}")
                rewards.append(0.5)
        return torch.tensor(rewards, dtype=torch.float32)

    def _score_combined(self, responses: List[str], ground_truths: List[str]) -> torch.Tensor:
        """
        Simple average of ROUGE-L and cosine similarity.

        Combining the two lightweight metrics partially compensates for their
        individual weaknesses: ROUGE-L favours sequential overlap while cosine
        similarity is order-agnostic, so together they give a more balanced
        signal without any model inference cost.
        """
        rouge_scores = self._score_rouge(responses, ground_truths)
        cosine_scores = self._score_cosine(responses, ground_truths)
        return (rouge_scores + cosine_scores) / 2.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_score(self, text: str) -> float:
        """Extract a numerical score in [0, 1] from LLM-generated text."""
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if match:
            score = float(match.group(1))
            return score if 0.0 <= score <= 1.0 else 0.5
        return 0.5