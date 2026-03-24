"""Test portfolio tracking and P&L calculations."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.portfolio import PortfolioState, compute_bip_fees, load_portfolio_state, save_portfolio_state, update_portfolio


class TestUpdatePortfolio:
    def test_zero_position_zero_return(self):
        """No position means no P&L (just fees on position change)."""
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(state, 0.0, 41000)

        assert metrics["btc_return_1h"] == pytest.approx(0.025)  # 2.5% BTC move
        assert metrics["fee_cost"] == 0.0  # no position change
        assert metrics["portfolio_return"] == 0.0  # no exposure
        assert new_state.portfolio_value == 1.0

    def test_long_position_positive_return(self):
        """Full long + price up = positive P&L."""
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(state, 1.0, 40400)  # +1%

        assert metrics["btc_return_1h"] == pytest.approx(0.01)
        assert metrics["fee_cost"] == 0.0  # no position change
        assert metrics["portfolio_return"] == pytest.approx(0.01)
        assert new_state.portfolio_value == pytest.approx(1.01)

    def test_short_position_price_down(self):
        """Short position + price down = positive P&L."""
        state = PortfolioState(position=-0.5, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(state, -0.5, 39600)  # -1%

        btc_ret = -0.01
        expected_return = -0.5 * btc_ret  # = +0.005
        assert metrics["portfolio_return"] == pytest.approx(expected_return)
        assert new_state.portfolio_value == pytest.approx(1.005)

    def test_position_change_fee(self):
        """Opening a position incurs fees."""
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(
            state, 1.0, 40000,  # price unchanged
            fee_rate=0.001, slippage_rate=0.0005,
        )

        assert metrics["position_delta"] == 1.0
        assert metrics["fee_cost"] == pytest.approx(0.0015)  # 15bps on full position
        # No BTC return (same price), but fee costs reduce portfolio
        assert new_state.portfolio_value == pytest.approx(1.0 - 0.0015)

    def test_partial_position_change_fee(self):
        """Resizing position: fee proportional to delta."""
        state = PortfolioState(position=0.3, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(state, 0.7, 40000)

        assert metrics["position_delta"] == pytest.approx(0.4)
        assert metrics["fee_cost"] == pytest.approx(0.4 * 0.0015)

    def test_drawdown_calculation(self):
        """Drawdown computed from peak."""
        state = PortfolioState(
            position=1.0, portfolio_value=1.05, peak_value=1.10, prev_btc_price=40000,
        )
        # Price drops 5%
        new_state, metrics = update_portfolio(state, 1.0, 38000)

        btc_ret = -0.05
        expected_value = 1.05 * (1 + 1.0 * btc_ret)  # = 1.05 * 0.95 = 0.9975
        expected_drawdown = (expected_value - 1.10) / 1.10

        assert new_state.portfolio_value == pytest.approx(expected_value)
        assert metrics["drawdown"] == pytest.approx(expected_drawdown)
        assert new_state.peak_value == 1.10  # peak unchanged

    def test_new_peak(self):
        """Peak updates when portfolio value exceeds it."""
        state = PortfolioState(
            position=1.0, portfolio_value=1.10, peak_value=1.10, prev_btc_price=40000,
        )
        new_state, metrics = update_portfolio(state, 1.0, 40800)  # +2%

        assert new_state.portfolio_value > 1.10
        assert new_state.peak_value == new_state.portfolio_value

    def test_first_run_no_prev_price(self):
        """First run with prev_btc_price=0 should have zero return."""
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=0.0)
        new_state, metrics = update_portfolio(state, 0.0, 40000)

        assert metrics["btc_return_1h"] == 0.0
        assert new_state.prev_btc_price == 40000

    def test_cumulative_pnl_sequence(self):
        """Known 3-step price sequence: verify cumulative P&L."""
        # Step 1: Enter long at 40000
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=40000)
        state, _ = update_portfolio(state, 1.0, 40000)  # Fee: 0.0015

        # Step 2: Price rises to 40400 (+1%)
        state, m2 = update_portfolio(state, 1.0, 40400)
        # Return: 1.0 * 0.01 - 0 = 0.01

        # Step 3: Close position at 40400
        state, m3 = update_portfolio(state, 0.0, 40400)
        # Return: 1.0 * 0 - 0.0015 = -0.0015

        # Expected: (1 - 0.0015) * (1 + 0.01) * (1 - 0.0015)
        expected = (1 - 0.0015) * (1 + 0.01) * (1 - 0.0015)
        assert state.portfolio_value == pytest.approx(expected, rel=1e-6)


class TestFundingRate:
    def test_long_positive_funding_is_cost(self):
        """Long + positive funding rate = cost (you pay)."""
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(
            state, 1.0, 40000, funding_rate=0.0008,  # per-8h
        )
        # hourly_rate = 0.0008 / 8 = 0.0001
        # funding_cost = 1.0 * 0.0001 = 0.0001
        assert metrics["funding_cost"] == pytest.approx(0.0001)
        assert metrics["portfolio_return"] == pytest.approx(-0.0001)  # no price change, just funding

    def test_short_positive_funding_is_credit(self):
        """Short + positive funding rate = credit (you receive)."""
        state = PortfolioState(position=-0.5, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(
            state, -0.5, 40000, funding_rate=0.0008,
        )
        # funding_cost = -0.5 * 0.0001 = -0.00005 (negative = credit)
        assert metrics["funding_cost"] == pytest.approx(-0.00005)
        assert metrics["portfolio_return"] == pytest.approx(0.00005)

    def test_flat_position_zero_funding(self):
        """Zero position = zero funding cost."""
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=40000)
        new_state, metrics = update_portfolio(
            state, 0.0, 40000, funding_rate=0.001,
        )
        assert metrics["funding_cost"] == 0.0

    def test_cumulative_funding_tracked(self):
        """Cumulative funding cost accumulates across runs."""
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=40000,
                               cumulative_funding_cost=0.001)
        new_state, _ = update_portfolio(state, 1.0, 40000, funding_rate=0.0008)
        assert new_state.cumulative_funding_cost == pytest.approx(0.001 + 0.0001)


class TestBipFees:
    def test_no_position_change(self):
        bip = compute_bip_fees(0.0, 70000)
        assert bip["n_contracts"] == 0
        assert bip["total_bip_cost"] == 0.0

    def test_full_position_change(self):
        """Full position change at $70k."""
        bip = compute_bip_fees(1.0, 70000, contract_size=0.01, fee_per_contract=0.46)
        # notional = 1.0 * 70000 = $70000
        # n_contracts = round(70000 / (0.01 * 70000)) = round(100) = 100
        assert bip["n_contracts"] == 100
        assert bip["fixed_fees"] == pytest.approx(46.0)  # 100 * $0.46
        assert bip["slippage"] == pytest.approx(70000 * 5.0 / 10000)  # 5bps of $70k

    def test_partial_position_change(self):
        bip = compute_bip_fees(0.1, 70000, contract_size=0.01, fee_per_contract=0.46)
        # notional = 0.1 * 70000 = $7000
        # n_contracts = round(7000 / 700) = 10
        assert bip["n_contracts"] == 10
        assert bip["fixed_fees"] == pytest.approx(4.6)


class TestPortfolioStatePersistence:
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = PortfolioState(
                position=0.5,
                portfolio_value=1.0234,
                peak_value=1.05,
                trade_count=42,
                prev_btc_price=67890.12,
            )
            save_portfolio_state(state, path)
            loaded = load_portfolio_state(path)

            assert loaded.position == pytest.approx(0.5)
            assert loaded.portfolio_value == pytest.approx(1.0234)
            assert loaded.peak_value == pytest.approx(1.05)
            assert loaded.trade_count == 42
            assert loaded.prev_btc_price == pytest.approx(67890.12)
        finally:
            os.unlink(path)

    def test_load_missing_returns_default(self):
        state = load_portfolio_state("/nonexistent/path.json")
        assert state.position == 0.0
        assert state.portfolio_value == 1.0
        assert state.trade_count == 0
