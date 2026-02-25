"""Smoke test: Verifies the full pipeline works end-to-end.

This script tests each component with synthetic data to make sure
all imports, configurations, and data flows work correctly.

Usage:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_config() -> bool:
    """Test configuration loading."""
    logger.info("--- Testing Config ---")
    from bci.utils.config import ExperimentConfig, load_config

    config = load_config()
    assert isinstance(config, ExperimentConfig)
    assert config.dataset.name == "bci_iv2a"
    assert config.model.vit_model_name == "vit_tiny_patch16_224"
    logger.info("  Config: OK (name=%s)", config.name)
    return True


def test_preprocessing() -> bool:
    """Test preprocessing pipeline with synthetic data."""
    logger.info("--- Testing Preprocessing ---")
    import mne

    from bci.data.preprocessing import PreprocessingPipeline
    from bci.utils.config import PreprocessingConfig

    config = PreprocessingConfig(apply_ica=False, resample_freq=128.0)
    pipeline = PreprocessingPipeline(config)

    # Create synthetic Raw object (3 channels, 10s, 256 Hz)
    n_channels, sfreq, duration = 3, 256.0, 10.0
    data = np.random.randn(n_channels, int(sfreq * duration)) * 1e-6
    ch_names = ["C3", "Cz", "C4"]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Test filtering
    raw_filtered = pipeline.filter(raw.copy())
    assert raw_filtered.info["sfreq"] == sfreq
    logger.info("  Filtering: OK")

    # Test resampling
    raw_resampled = pipeline.resample(raw_filtered.copy())
    assert raw_resampled.info["sfreq"] == config.resample_freq
    logger.info("  Resampling: OK (%.0f -> %.0f Hz)", sfreq, config.resample_freq)

    return True


def test_cwt_transform() -> bool:
    """Test CWT spectrogram generation."""
    logger.info("--- Testing CWT Spectrogram ---")
    from bci.data.transforms import CWTSpectrogramTransform
    from bci.utils.config import SpectrogramConfig

    config = SpectrogramConfig(
        n_freqs=32,
        image_size=(64, 64),  # Small for testing
        channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(config)

    # Synthetic epoch data: 5 trials, 3 channels, 512 time points
    n_trials, n_channels, n_times = 5, 3, 512
    sfreq = 128.0
    X = np.random.randn(n_trials, n_channels, n_times)
    channel_names = ["C3", "Cz", "C4"]

    images = transform.transform_epochs(X, channel_names, sfreq)
    assert images.shape == (n_trials, 64, 64, 3)
    assert images.dtype == np.uint8
    logger.info("  CWT Transform: OK (shape=%s)", images.shape)

    return True


def test_csp_features() -> bool:
    """Test CSP feature extraction."""
    logger.info("--- Testing CSP Features ---")
    from bci.features.csp import CSPFeatureExtractor

    csp = CSPFeatureExtractor(n_components=3)

    # Synthetic: 40 trials, 8 channels, 256 time points, 2 classes
    n_trials, n_channels, n_times = 40, 8, 256
    X = np.random.randn(n_trials, n_channels, n_times)
    y = np.array([0] * 20 + [1] * 20)

    features = csp.fit_transform(X, y)
    assert features.shape == (n_trials, 3)  # n_components
    logger.info("  CSP Features: OK (shape=%s)", features.shape)

    return True


def test_riemannian_features() -> bool:
    """Test Riemannian feature extraction."""
    logger.info("--- Testing Riemannian Features ---")
    from bci.features.riemannian import RiemannianFeatureExtractor

    riemann = RiemannianFeatureExtractor(estimator="scm", metric="riemann")

    # Synthetic: 40 trials, 8 channels, 256 time points
    n_trials, n_channels, n_times = 40, 8, 256
    X = np.random.randn(n_trials, n_channels, n_times)
    y = np.array([0] * 20 + [1] * 20)

    features = riemann.fit_transform(X, y)
    expected_dim = n_channels * (n_channels + 1) // 2  # 36
    assert features.shape == (n_trials, expected_dim)
    logger.info("  Riemannian Features: OK (shape=%s)", features.shape)

    return True


def test_vit_branch() -> bool:
    """Test ViT branch forward pass."""
    logger.info("--- Testing ViT Branch ---")
    from bci.models.vit_branch import ViTBranch
    from bci.utils.config import ModelConfig

    config = ModelConfig(vit_pretrained=False)  # Skip download for test
    model = ViTBranch(config=config, as_feature_extractor=True)

    # Synthetic batch: 2 images, 3 channels, 224x224
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model(x)

    assert features.shape == (2, model.feature_dim)
    logger.info(
        "  ViT Branch: OK (output=%s, params=%dk)",
        features.shape, model.get_num_params(trainable_only=False) // 1000,
    )

    return True


def test_math_branch() -> bool:
    """Test Math branch forward pass."""
    logger.info("--- Testing Math Branch ---")
    from bci.models.math_branch import MathBranch
    from bci.utils.config import ModelConfig

    config = ModelConfig()
    input_dim = 48  # e.g., 12 CSP + 36 Riemannian
    model = MathBranch(input_dim=input_dim, config=config)

    x = torch.randn(4, input_dim)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (4, model.output_dim)
    logger.info("  Math Branch: OK (output=%s)", output.shape)

    return True


def test_dual_branch() -> bool:
    """Test the full dual-branch model."""
    logger.info("--- Testing Dual Branch Model ---")
    from bci.models.dual_branch import DualBranchModel
    from bci.utils.config import ModelConfig

    config = ModelConfig(vit_pretrained=False)
    math_input_dim = 48
    model = DualBranchModel(math_input_dim=math_input_dim, config=config)

    images = torch.randn(2, 3, 224, 224)
    features = torch.randn(2, math_input_dim)

    with torch.no_grad():
        logits = model(images, features)

    assert logits.shape == (2, config.n_classes)
    logger.info(
        "  Dual Branch: OK (logits=%s, total_params=%dk)",
        logits.shape,
        sum(p.numel() for p in model.parameters()) // 1000,
    )

    return True


def test_datasets() -> bool:
    """Test PyTorch dataset classes."""
    logger.info("--- Testing PyTorch Datasets ---")
    from bci.data.dataset import DualBranchDataset, EEGFeatureDataset, SpectrogramDataset

    n = 10

    # SpectrogramDataset
    images = np.random.randint(0, 255, (n, 64, 64, 3), dtype=np.uint8)
    labels = np.array([0, 1] * 5)
    ds = SpectrogramDataset(images, labels)
    img, lbl = ds[0]
    assert img.shape == (3, 64, 64)
    logger.info("  SpectrogramDataset: OK")

    # EEGFeatureDataset
    features = np.random.randn(n, 48).astype(np.float32)
    ds2 = EEGFeatureDataset(features, labels)
    feat, lbl = ds2[0]
    assert feat.shape == (48,)
    logger.info("  EEGFeatureDataset: OK")

    # DualBranchDataset
    ds3 = DualBranchDataset(images, features, labels)
    img, feat, lbl = ds3[0]
    assert img.shape == (3, 64, 64)
    assert feat.shape == (48,)
    logger.info("  DualBranchDataset: OK")

    return True


def test_device() -> bool:
    """Test device detection."""
    logger.info("--- Testing Device Utils ---")
    from bci.utils.seed import get_device, set_seed

    set_seed(42)
    device = get_device("auto")
    logger.info("  Device: %s", device)

    return True


def main() -> None:
    """Run all smoke tests."""
    tests = [
        ("Config", test_config),
        ("Preprocessing", test_preprocessing),
        ("CWT Transform", test_cwt_transform),
        ("CSP Features", test_csp_features),
        ("Riemannian Features", test_riemannian_features),
        ("ViT Branch", test_vit_branch),
        ("Math Branch", test_math_branch),
        ("Dual Branch Model", test_dual_branch),
        ("PyTorch Datasets", test_datasets),
        ("Device Utils", test_device),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            logger.error("FAILED: %s - %s", name, e)
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", status, name)
        if not passed:
            all_passed = False

    logger.info("=" * 50)
    if all_passed:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
