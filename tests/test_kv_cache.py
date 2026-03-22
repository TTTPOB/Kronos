import numpy as np
import pytest
import torch

from model import Kronos, KronosTokenizer
from model.kronos import _auto_regressive_inference_no_cache, auto_regressive_inference


def _build_tiny_model_pair():
    tokenizer = KronosTokenizer(
        d_in=6,
        d_model=16,
        n_heads=4,
        ff_dim=32,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        s1_bits=2,
        s2_bits=2,
        beta=0.1,
        gamma0=0.1,
        gamma=0.1,
        zeta=0.1,
        group_size=2,
    )
    model = Kronos(
        s1_bits=2,
        s2_bits=2,
        n_layers=2,
        d_model=16,
        n_heads=4,
        ff_dim=32,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )
    tokenizer.eval()
    model.eval()
    return tokenizer, model


def _build_inference_inputs(batch_size=2, seq_len=6, pred_len=3):
    x = torch.randn(batch_size, seq_len, 6)
    x_stamp = torch.zeros(batch_size, seq_len, 5)
    y_stamp = torch.zeros(batch_size, pred_len, 5)
    for stamp in (x_stamp, y_stamp):
        stamp[:, :, 0] = torch.randint(0, 60, stamp[:, :, 0].shape)
        stamp[:, :, 1] = torch.randint(0, 24, stamp[:, :, 1].shape)
        stamp[:, :, 2] = torch.randint(0, 7, stamp[:, :, 2].shape)
        stamp[:, :, 3] = torch.randint(1, 32, stamp[:, :, 3].shape)
        stamp[:, :, 4] = torch.randint(1, 13, stamp[:, :, 4].shape)
    return x, x_stamp, y_stamp


def test_prefill_and_incremental_decode_match_full_decode():
    torch.manual_seed(123)
    _, model = _build_tiny_model_pair()

    prompt_pre = torch.randint(0, 4, (2, 3))
    prompt_post = torch.randint(0, 4, (2, 3))
    _, prompt_stamp, _ = _build_inference_inputs(batch_size=2, seq_len=3, pred_len=1)
    prompt_stamp = prompt_stamp[:, :3, :]

    cached_logits, _, layer_caches = model.prefill_for_generation(
        prompt_pre,
        prompt_post,
        prompt_stamp,
        max_cache_len=4,
        positions=torch.arange(3),
    )
    full_logits, _ = model.decode_s1(prompt_pre, prompt_post, prompt_stamp)
    torch.testing.assert_close(cached_logits, full_logits[:, -1, :], rtol=0.0, atol=1e-6)

    next_pre = torch.randint(0, 4, (2, 1))
    next_post = torch.randint(0, 4, (2, 1))
    _, _, next_stamp = _build_inference_inputs(batch_size=2, seq_len=1, pred_len=1)

    next_logits, new_hidden, _ = model.decode_next_for_generation(
        next_pre,
        next_post,
        layer_caches,
        stamp=next_stamp,
        position=[3],
        max_cache_len=4,
    )
    expected_pre = torch.cat([prompt_pre, next_pre], dim=1)
    expected_post = torch.cat([prompt_post, next_post], dim=1)
    expected_stamp = torch.cat([prompt_stamp, next_stamp], dim=1)
    full_next_logits, full_next_hidden = model.decode_s1(
        expected_pre,
        expected_post,
        expected_stamp,
    )
    torch.testing.assert_close(next_logits, full_next_logits[:, -1, :], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(new_hidden, full_next_hidden[:, -1:, :], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("seq_len", "pred_len", "max_context"),
    [
        (3, 2, 8),
        (6, 3, 4),
    ],
)
@pytest.mark.parametrize("return_samples", [False, True])
def test_auto_regressive_inference_matches_uncached(seq_len, pred_len, max_context, return_samples):
    torch.manual_seed(321)
    tokenizer, model = _build_tiny_model_pair()
    x, x_stamp, y_stamp = _build_inference_inputs(seq_len=seq_len, pred_len=pred_len)

    kwargs = dict(
        tokenizer=tokenizer,
        model=model,
        x=x,
        x_stamp=x_stamp,
        y_stamp=y_stamp,
        max_context=max_context,
        pred_len=pred_len,
        clip=5,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=3,
        verbose=False,
        return_samples=return_samples,
        autocast_dtype=None,
    )

    torch.manual_seed(2026)
    uncached = _auto_regressive_inference_no_cache(**kwargs)
    torch.manual_seed(2026)
    cached = auto_regressive_inference(**kwargs)

    np.testing.assert_allclose(cached, uncached, rtol=0.0, atol=1e-6)
