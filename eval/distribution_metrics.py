"""Frozen standard distribution-metric evaluator v1.0.

Uses the Guo et al. 2022 pretrained contrastive text/motion encoders
(EvaluatorMDMWrapper) to compute FID, R-Precision, Diversity, MM-Dist
over saved motion files.

All motion data passed to compute_*() must be in MDM-normalized space
(as saved in comparison.npy motion_hml / motion_hml_tj).
"""

import sys
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DistributionMetrics:

    def __init__(self, device="cuda"):
        self.device = device
        self._eval_wrapper = None
        self._word_vectorizer = None
        self.ref_mu = None
        self.ref_cov = None
        self._cache_path = _PROJECT_ROOT / "eval" / "assets" / "ref_stats.npz"

        mdm_root = _PROJECT_ROOT / "dataset" / "HumanML3D"
        self._mean = np.load(mdm_root / "Mean.npy").astype(np.float32)
        self._std = np.load(mdm_root / "Std.npy").astype(np.float32)

    # ---- Lazy loaders ----

    @property
    def eval_wrapper(self):
        if self._eval_wrapper is None:
            from data_loaders.humanml.networks.evaluator_wrapper import \
                EvaluatorMDMWrapper
            self._eval_wrapper = EvaluatorMDMWrapper("humanml", self.device)
        return self._eval_wrapper

    @property
    def word_vectorizer(self):
        if self._word_vectorizer is None:
            from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
            self._word_vectorizer = WordVectorizer(
                str(_PROJECT_ROOT / "glove"), "our_vab")
        return self._word_vectorizer

    # ---- Reference stats (precompute once, cache) ----

    def ensure_ref_stats(self, force=False):
        if (not force) and self._cache_path.exists():
            data = np.load(self._cache_path)
            self.ref_mu = data["mu"]
            self.ref_cov = data["cov"]
            return

        print("[eval] Computing reference statistics from HumanML3D test set ...")
        from data_loaders.humanml.utils.metrics import \
            calculate_activation_statistics

        all_emb, batch_motions, batch_lens = [], [], []
        test_file = _PROJECT_ROOT / "dataset" / "HumanML3D" / "test.txt"
        vecs_dir = _PROJECT_ROOT / "dataset" / "HumanML3D" / "new_joint_vecs"

        with open(test_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        for name in lines:
            m = np.load(vecs_dir / f"{name}.npy")          # raw scale
            # Normalize to MDM space (same as evaluator training)
            m_norm = (m - self._mean) / self._std
            batch_motions.append(m_norm.astype(np.float32))
            batch_lens.append(m.shape[0])
            if len(batch_motions) >= 64:
                emb = self._encode_motion_batch(batch_motions, batch_lens)
                all_emb.append(emb)
                batch_motions, batch_lens = [], []

        if batch_motions:
            emb = self._encode_motion_batch(batch_motions, batch_lens)
            all_emb.append(emb)

        all_emb = np.concatenate(all_emb, axis=0)
        self.ref_mu, self.ref_cov = calculate_activation_statistics(all_emb)

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._cache_path, mu=self.ref_mu, cov=self.ref_cov)
        print(f"[eval] Reference stats cached -> {self._cache_path}  "
              f"(N={all_emb.shape[0]}, dim={all_emb.shape[1]})")

    # ---- Motion encoding ----

    def _encode_motion_batch(self, motions, m_lens):
        import torch
        max_len = max(m.shape[0] for m in motions)
        B = len(motions)
        padded = np.zeros((B, max_len, 263), dtype=np.float32)
        for i, m in enumerate(motions):
            padded[i, :m.shape[0]] = m
        m_t = torch.from_numpy(padded).to(self.device)
        l_t = torch.as_tensor(m_lens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb = self.eval_wrapper.get_motion_embeddings(m_t, l_t)
        return emb.cpu().numpy()

    def _pad_motions_to_tensor(self, motions):
        import torch
        if isinstance(motions, np.ndarray):
            if motions.ndim == 3:
                return torch.from_numpy(motions).float().to(self.device)
            return torch.from_numpy(motions[None]).float().to(self.device)
        max_len = max(m.shape[0] for m in motions)
        B = len(motions)
        padded = np.zeros((B, max_len, 263), dtype=np.float32)
        for i, m in enumerate(motions):
            padded[i, :m.shape[0]] = m
        return torch.from_numpy(padded).to(self.device)

    # ---- Public metrics (all expect MDM-normalized input) ----

    def compute_fid(self, motions, m_lens):
        if self.ref_mu is None:
            self.ensure_ref_stats()
        from data_loaders.humanml.utils.metrics import \
            calculate_activation_statistics
        emb = self._encode_motion_batch(motions, m_lens)
        mu, cov = calculate_activation_statistics(emb)
        return self._compute_fid_stable(self.ref_mu, self.ref_cov, mu, cov)

    def compute_diversity(self, motions, m_lens, times=300):
        from data_loaders.humanml.utils.metrics import calculate_diversity
        emb = self._encode_motion_batch(motions, m_lens)
        n = emb.shape[0]
        t = min(times, max(1, n // 2))
        return float(calculate_diversity(emb, t))

    def compute_r_precision(self, motions, m_lens, texts, k=3):
        import torch
        from data_loaders.humanml.utils.metrics import calculate_R_precision
        m_t = self._pad_motions_to_tensor(motions)
        l_t = torch.as_tensor(m_lens, dtype=torch.long, device=self.device)
        we, po, cl = self._texts_to_batch(texts)
        with torch.no_grad():
            te, me = self.eval_wrapper.get_co_embeddings(we, po, cl, m_t, l_t)
        top_k_mat = calculate_R_precision(
            te.cpu().numpy(), me.cpu().numpy(), k, sum_all=True)
        return (top_k_mat / me.shape[0]).tolist()

    def compute_mm_dist(self, motions, m_lens, texts):
        import torch
        from data_loaders.humanml.utils.metrics import calculate_matching_score
        m_t = self._pad_motions_to_tensor(motions)
        l_t = torch.as_tensor(m_lens, dtype=torch.long, device=self.device)
        we, po, cl = self._texts_to_batch(texts)
        with torch.no_grad():
            te, me = self.eval_wrapper.get_co_embeddings(we, po, cl, m_t, l_t)
        mm = calculate_matching_score(
            te.cpu().numpy(), me.cpu().numpy(), sum_all=True)
        return float(mm) / me.shape[0]

    # ---- Text helpers ----

    def _texts_to_batch(self, texts):
        import torch
        we_list, po_list, cl = [], [], []
        for raw in texts:
            tokens = self._tokenize(raw)
            we, po = [], []
            for token in tokens:
                w, p = self.word_vectorizer[token]
                we.append(w)
                po.append(p)
            we_list.append(np.stack(we))
            po_list.append(np.stack(po))
            cl.append(len(we))
        max_len = max(cl)
        B = len(texts)
        we_b = np.zeros((B, max_len, 300), dtype=np.float32)
        po_b = np.zeros((B, max_len, 15), dtype=np.float32)
        for i in range(B):
            we_b[i, :cl[i]] = we_list[i]
            po_b[i, :cl[i]] = po_list[i]
        return (torch.from_numpy(we_b).to(self.device),
                torch.from_numpy(po_b).to(self.device),
                torch.as_tensor(cl, dtype=torch.long, device=self.device))

    @staticmethod
    def _tokenize(sentence):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(sentence.replace("-", " "))
            tokens = []
            for tok in doc:
                if not tok.text.strip():
                    continue
                word = (tok.lemma_ if (tok.pos_ in ("NOUN", "VERB")
                        and tok.text != "left") else tok.text)
                pos_map = {"NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ",
                           "ADV": "ADV", "ADP": "ADP", "DET": "DET",
                           "NUM": "NUM", "PRON": "PRON", "AUX": "AUX"}
                pos = pos_map.get(tok.pos_, "OTHER")
                tokens.append(f"{word}/{pos}")
            return ["sos/OTHER"] + tokens + ["eos/OTHER"]
        except Exception:
            words = sentence.split()
            return ["sos/OTHER"] + [f"{w}/NOUN" for w in words] + ["eos/OTHER"]

    # ---- FID numerical stability ----

    @staticmethod
    def _compute_fid_stable(mu1, sigma1, mu2, sigma2):
        import scipy.linalg as linalg
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        d = sigma1.shape[0]
        tr1 = np.trace(sigma1) / d
        tr2 = np.trace(sigma2) / d
        ridge = max(tr1, tr2) * 1e-3 + 1e-6
        s1 = sigma1 + np.eye(d) * ridge
        s2 = sigma2 + np.eye(d) * ridge
        try:
            covmean = linalg.sqrtm(s1.dot(s2), disp=False)[0]
        except Exception:
            covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean
        return float(max(0.0, fid))