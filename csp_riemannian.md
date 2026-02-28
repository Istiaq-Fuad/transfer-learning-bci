## 1. CSP + LDA Baseline Improvements

### Current Performance

- Within-subject: ~77% (±12.53%, κ=0.586)
- LOSO: ~60% (±10.84%, κ=0.316)
- Analysis: Solid but limited by single-band filtering and no cross-subject adaptation. SOTA CSP variants reach 82-85% within-subject (e.g., FBCSP + SVM ~83% in [Ang et al., 2008](https://ieeexplore.ieee.org/document/4634130)).

### Suggested Changes

#### Quick Wins

1. **Filter Bank CSP (FBCSP)**: Apply CSP across multiple frequency bands for better mu/beta rhythm capture.
2. **Advanced Regularization**: Add shrinkage to covariance estimation.
3. **Ensemble CSP**: Train on multiple time windows and ensemble predictions.

#### Longer-Term

1. **Subject-Specific CSP (SSCSP)**: Align filters across subjects for better LOSO.

### Where to Make Changes

- **Core Feature Extractor**: `src/bci/features/csp.py` (CSPFeatureExtractor class).
- **Config**: `configs/default.yaml` (add `bands` and `time_windows` params).
- **Script**: `scripts/baseline_a_csp_lda.py` (update `predict_fn` for ensemble).
- **Pipeline Integration**: `src/bci/data/dual_branch_builder.py` (for fusion use).

### How to Implement

1. **FBCSP**:

   - Add `bands` to config: `bands: [[4,8], [8,12], [12,16], [16,20], [20,24], [24,28], [28,32], [32,36], [36,40]]`.
   - In `CSPFeatureExtractor.__init__`, add `self.bands = bands or [[4,40]]`.
   - In `.fit_transform`, loop over bands:
     ```python
     from mne.filter import filter_data
     feats = []
     for l_freq, h_freq in self.bands:
         X_band = filter_data(X, sfreq=128.0, l_freq=l_freq, h_freq=h_freq)
         csp_band = CSP(n_components=self.n_components, reg=self.reg)
         feats_band = csp_band.fit_transform(X_band, y)
         feats.append(feats_band)
     return np.hstack(feats)  # Shape: (n_trials, n_bands * n_components)
     ```
   - Feature selection: After concat, use `SelectKBest(score_func=mutual_info_classif, k=20)`.

2. **Advanced Regularization**:

   - In `.fit`, before CSP:
     ```python
     from pyriemann.estimation import covariances
     covs = covariances(X, estimator='scm', shrinkage=0.1)  # Or 'oas'
     self.csp.fit(covs, y)  # Note: CSP accepts covs directly if using MNE's CSP
     ```

3. **Ensemble CSP**:

   - Add `time_windows: [[0,2], [1,3], [2,4]]` to config (in seconds, assuming tmin=0, tmax=4).
   - In `predict_fn`, create list of CSPs:
     ```python
     preds, probs = [], []
     for tmin, tmax in time_windows:
         idx_start, idx_end = int(tmin * 128), int(tmax * 128)
         X_train_win = X_train[:, :, idx_start:idx_end]
         X_test_win = X_test[:, :, idx_start:idx_end]
         csp = CSPFeatureExtractor(...)  # Fit on window
         feats_train = csp.fit_transform(X_train_win, y_train)
         feats_test = csp.transform(X_test_win)
         lda.fit(feats_train, y_train)
         preds.append(lda.predict(feats_test))
         probs.append(lda.predict_proba(feats_test))
     y_pred = mode(np.array(preds), axis=0)[0]  # Voting
     y_prob = np.mean(probs, axis=0)  # Average probs
     ```

4. **SSCSP**:
   - In LOSO loop (`loso_cv`), for each held-out subject: Compute Riemannian mean of source covs, align target covs via `pyriemann.preprocessing.align_recenter`.

### Expected Gains

- Quick Wins: +5-8% within-subject (82-85%), +3-5% LOSO (63-65%).
- Longer-Term: +5-10% LOSO (65-70%).

---

## 2. Riemannian + LDA Baseline Improvements

### Current Performance

- Within-subject/LOSO: ~62% (κ=0.233/0.277)
- Analysis: Low due to basic tangent space without bands or alignment. SOTA Riemannian reaches 82-87% within-subject (e.g., ATL-RM ~84.5% in [Ng et al., 2021](https://ieeexplore.ieee.org/document/9433745)).

### Suggested Changes

#### Quick Wins

1. **Filter Bank Tangent Space (FBTS)**: Tangent space per band.
2. **Better Covariance**: Switch estimator, add extended covs.
3. **Riemannian Alignment (RA)**: Recenter for cross-subject.

#### Longer-Term (Recommended with Code)

1. **Adaptive Transfer Learning with Riemannian Manifold (ATL-RM)**: From 2021 paper, code available.

### Where to Make Changes

- **Core Feature Extractor**: `src/bci/features/riemannian.py` (RiemannianFeatureExtractor).
- **Config**: `configs/default.yaml` (add `bands`, `estimator='oas'`).
- **Script**: `scripts/baseline_b_riemannian.py` (update `predict_fn` for FBTS/RA).
- **Pipeline Integration**: `src/bci/data/dual_branch_builder.py` (for fusion).

### How to Implement

1. **FBTS**:

   - Similar to FBCSP: Loop over bands in `.fit_transform`:
     ```python
     from pyriemann.tangentspace import TangentSpace
     feats = []
     for l_freq, h_freq in self.bands:
         X_band = filter_data(X, sfreq=128.0, l_freq=l_freq, h_freq=h_freq)
         covs = covariances(X_band, estimator=self.estimator)
         ts = TangentSpace(metric='riemann')
         feats_band = ts.fit_transform(covs, y)
         feats.append(feats_band)
     return np.hstack(feats)  # Higher dim, use PCA if needed
     ```

2. **Better Covariance**:

   - In `__init__`: `self.estimator = 'oas'` (or 'ts').
   - For extended: Use `pyriemann.estimation.XdawnCovariances` if adding supervision.

3. **RA**:

   - In `predict_fn` for LOSO:
     ```python
     from pyriemann.utils.mean import mean_riemann
     from pyriemann.preprocessing import align_recenter
     # Compute ref_cov = mean_riemann(train_covs) from source subjects
     covs_train_aligned = align_recenter(train_covs, ref_cov)
     covs_test_aligned = align_recenter(test_covs, ref_cov)
     # Then fit ts on aligned
     ```

4. **ATL-RM (Advanced with Code)**:
   - Clone repo: https://github.com/tonyngjichun/ATL-RM
   - Extract ATL module: Create new class in `riemannian.py`:
     ```python
     from atl_rm import ATL  # Assume you add to src
     class AdvancedRiemannian(ATL):
         def fit_transform(self, X, y):
             # Call ATL's align and project
             covs = self.compute_cov(X)
             aligned = self.align(covs)
             return self.tangent_space(aligned, y)
     ```
   - Replace in `predict_fn`: Use `AdvancedRiemannian()` instead of basic.
   - Adapt their logistic head if LDA underperforms.

### Expected Gains

- Quick Wins: +6-10% (68-72% both).
- ATL-RM: +15-20% (77-82% within, 70-75% LOSO).
