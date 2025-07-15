import numpy as np
import pandas as pd
import sys
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from datetime import timedelta

class PairsTradingPortfolio:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital  # For portfolio construction
        self.strategy_initial_capital = 1  # Normalized to 1 for percentage returns in strategy
        self.rf_rates = {
            2007: 0.00033, 2008: 0.00033, 2009: 0.00033, 
            2010: 0.00033, 2011: 0.00033, 2012: 0.00033,
            2013: 0.00033, 2014: 0.00033, 2015: 0.00053,
            2016: 0.0032, 2017: 0.0093, 2018: 0.0194
        }

    # Strategy methods
    def apply_strategy(self, pairs: List[Tuple[str, str, Dict]], 
                      strategy: str = 'fixed_beta',
                      entry_z: float = 1.5, 
                      exit_z: float = 0.5,
                      test_mode: bool = False,
                      train_val_split: str = '2010-01-01') -> Tuple:
        """
        Main method to apply trading strategy to all pairs
        """
        sharpe_results = []
        cum_returns = []
        sharpe_results_with_costs = []
        cum_returns_with_costs = []
        performance = []
        
        print("Initializing pairs trading strategy...")
        
        for i, pair in enumerate(pairs):
            sys.stdout.write(f"\rProcessing pair {i+1}/{len(pairs)}")
            sys.stdout.flush()
            
            pair_info = pair[2]
            y = pair_info['Y_test'] if test_mode else pair_info['Y_train'][train_val_split:]
            x = pair_info['X_test'] if test_mode else pair_info['X_train'][train_val_split:]

            if strategy == 'fixed_beta':
                summary, sharpe, balance_summary = self.threshold_strategy(
                    y=y, x=x, beta=pair_info['coint_coef'],
                    entry_level=entry_z,
                    exit_level=exit_z
                )
                
                cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
                sharpe_results.append(sharpe[0])
                cum_returns_with_costs.append((summary.account_balance.iloc[-1] - 1) * 100)
                sharpe_results_with_costs.append(sharpe[1])
                performance.append((pair, summary, balance_summary))
            else:
                raise ValueError("Only 'fixed_beta' strategy is currently implemented")

        return (sharpe_results, cum_returns), (sharpe_results_with_costs, cum_returns_with_costs), performance

    def threshold_strategy(self, y, x, beta, entry_level=1.0, exit_level=1.0, stabilization=5):
        """Implements fixed beta threshold strategy."""
        spread = y - beta * x
        norm_spread = (spread - spread.mean()) / np.std(spread)
        norm_spread = np.asarray(norm_spread.values)

        # Generate trading signals
        longs_entry = norm_spread < -entry_level
        longs_exit = norm_spread > -exit_level 
        shorts_entry = norm_spread > entry_level
        shorts_exit = norm_spread < exit_level

        num_units_long = pd.Series(np.nan, index=y.index)
        num_units_short = pd.Series(np.nan, index=y.index)

        # Stabilization period
        longs_entry[:stabilization] = False
        longs_exit[:stabilization] = False
        shorts_entry[:stabilization] = False
        shorts_exit[:stabilization] = False

        # Set positions
        num_units_long[longs_entry] = 1.
        num_units_long[longs_exit] = 0
        num_units_short[shorts_entry] = -1.
        num_units_short[shorts_exit] = 0

        # Process positions
        num_units_long = num_units_long.shift(1).fillna(0)
        num_units_short = num_units_short.shift(1).fillna(0)
        num_units = (num_units_long + num_units_short).rename('numUnits')

        trading_durations = self._add_trading_duration(pd.DataFrame(num_units, index=y.index))

        # Calculate returns
        position_ret, _, ret_summary = self.calculate_returns(y, x, beta, num_units)
        balance_summary = self.calculate_balance(y, x, beta, num_units.shift(1).fillna(0), trading_durations)

        # Compile results
        summary_data = [
            (balance_summary.pnl, 'pnl'),
            (balance_summary.pnl_y, 'pnl_y'),
            (balance_summary.pnl_x, 'pnl_x'),
            (balance_summary.account_balance, 'account_balance'),
            (balance_summary.returns, 'returns'),
            (position_ret, 'position_return'),
            (y, y.name),
            (x, x.name),
            (pd.Series(norm_spread, index=y.index), 'norm_spread'),
            (num_units, 'numUnits'),
            (trading_durations, 'trading_duration')
        ]
        summary = self.trade_summary(summary_data, beta)

        # Calculate Sharpe ratios
        n_years = round(len(y) / 240)
        sharpe_no_costs = 0 if np.std(position_ret) == 0 else self._calculate_sharpe(n_years, 252, position_ret)
        sharpe_w_costs = 0 if np.std(summary.returns) == 0 else self._calculate_sharpe(n_years, 252, summary.returns)

        return summary, (sharpe_no_costs, sharpe_w_costs), balance_summary

    def _add_trading_duration(self, df):
        """Calculates trade durations in days."""
        df['trading_duration'] = 0
        prev_unit = 0.
        counter = 0
        current_day = df.index[0].day
        
        for idx, row in df.iterrows():
            if prev_unit == row['numUnits']:
                if prev_unit != 0.:
                    if idx.day != current_day:
                        counter += 1
                        current_day = idx.day
                    if idx == df.index[-1]:
                        df.loc[idx, 'trading_duration'] = counter
                continue
                
            if prev_unit == 0.:
                prev_unit = row['numUnits']
                counter = 1
                current_day = idx.day
            else:
                df.loc[idx, 'trading_duration'] = counter
                prev_unit = row['numUnits']
                counter = 1
                current_day = idx.day
                
        return df['trading_duration']

    def calculate_returns(self, y, x, beta, positions):
        """Calculates position returns without costs."""
        y = y.copy().rename('y')
        x = x.copy().rename('x')
        
        new_positions = positions.diff()[positions.diff() != 0].index.values
        end_position = pd.Series(0, index=y.index, name='end_position')
        end_position[new_positions] = 1.
        if positions.iloc[-1] != 0:
            end_position.iloc[-1] = 1.

        y_entry = pd.Series(np.nan, index=y.index, name='y_entry')
        x_entry = pd.Series(np.nan, index=y.index, name='x_entry')
        y_entry[new_positions] = y[new_positions]
        x_entry[new_positions] = x[new_positions]
        y_entry = y_entry.shift().ffill()
        x_entry = x_entry.shift().ffill()

        positions = positions.rename('positions')
        df = pd.concat([y, x, positions.shift().fillna(0), y_entry, x_entry, end_position], axis=1)
        
        returns = df.apply(lambda row: self._return_per_position(row, beta), axis=1).fillna(0)
        cum_returns = np.cumprod(returns + 1) - 1
        df['ret'] = returns
        returns.name = 'position_return'

        return returns, cum_returns, df

    def _return_per_position(self, row, beta=None):
        if row['end_position'] != 0:
            y_ret = (row.iloc[0] - row['y_entry']) / row['y_entry']
            x_ret = (row.iloc[1] - row['x_entry']) / row['x_entry']
            if beta > 1.:
                return ((1 / beta) * y_ret - x_ret) * row['positions']
            return (y_ret - beta * x_ret) * row['positions']
        return 0

    def calculate_balance(self, y, x, beta, positions, durations):
        """Tracks portfolio balance with costs."""
        y_ret = y.pct_change().fillna(0) * positions
        x_ret = -x.pct_change().fillna(0) * positions

        leg_y = np.full(len(y), np.nan)
        leg_x = np.full(len(y), np.nan)
        pnl_y = np.full(len(y), np.nan)
        pnl_x = np.full(len(y), np.nan)
        balance = np.full(len(y), np.nan)

        # Position triggers
        new_pos_idx = positions.diff()[positions.diff() != 0].index.values
        end_pos_idx = durations[durations != 0].index.values
        triggers = pd.Series(0, index=y.index, name='position_trigger')
        triggers[new_pos_idx] = 2.
        triggers[end_pos_idx] -= 1.
        triggers = triggers * positions.abs()

        for i in range(len(y)):
            if i == 0:
                pnl_y[0] = pnl_x[0] = 0
                balance[0] = 1
                leg_y[0] = 1/beta if beta > 1 else 1
                leg_x[0] = 1 if beta > 1 else beta
            elif positions.iloc[i] == 0:
                pnl_y[i] = pnl_x[i] = 0
                leg_y[i] = leg_y[i-1]
                leg_x[i] = leg_x[i-1]
                balance[i] = balance[i-1]
            else:
                self._process_position(
                    i, y_ret, x_ret, beta, positions, triggers, durations,
                    leg_y, leg_x, pnl_y, pnl_x, balance
                )

        return self._compile_balance_results(y, x, balance, pnl_y, pnl_x, leg_y, leg_x, triggers, positions, durations)

    def _process_position(self, i, y_ret, x_ret, beta, positions, triggers, durations,
                         leg_y, leg_x, pnl_y, pnl_x, balance):
        """Helper method to process trading positions."""
        trigger = triggers.iloc[i]
        prev_bal = balance[i-1]
        
        if trigger == 1:  # New single-day position
            self._process_new_position(i, y_ret, x_ret, beta, positions, prev_bal, 
                                     leg_y, leg_x, pnl_y, pnl_x)
            self._apply_costs(i, beta, positions, durations, pnl_y, pnl_x, prev_bal, is_new=True)
            
        elif trigger == 2:  # New multi-day position 
            self._process_new_position(i, y_ret, x_ret, beta, positions, prev_bal,
                                     leg_y, leg_x, pnl_y, pnl_x)
            self._apply_costs(i, beta, positions, durations, pnl_y, pnl_x, prev_bal)
            
        else:  # Ongoing position
            pnl_y[i] = y_ret.iloc[i] * leg_y[i-1]
            pnl_x[i] = x_ret.iloc[i] * leg_x[i-1]
            
            if positions.iloc[i] == 1:
                leg_y[i] = leg_y[i-1] + pnl_y[i]
                leg_x[i] = leg_x[i-1] - pnl_x[i]
            else:
                leg_y[i] = leg_y[i-1] - pnl_y[i]
                leg_x[i] = leg_x[i-1] + pnl_x[i]
                
            if trigger == -1:  # Position ending
                self._apply_short_costs(i, beta, positions, durations, pnl_y, pnl_x, prev_bal)
                
        balance[i] = balance[i-1] + pnl_y[i] + pnl_x[i]

    def _process_new_position(self, i, y_ret, x_ret, beta, positions, prev_bal, 
                            leg_y, leg_x, pnl_y, pnl_x):
        """Processes new position setup."""
        if beta > 1:
            pnl_y[i] = y_ret.iloc[i] * prev_bal * (1/beta)
            pnl_x[i] = x_ret.iloc[i] * prev_bal
        else:
            pnl_y[i] = y_ret.iloc[i] * prev_bal
            pnl_x[i] = x_ret.iloc[i] * prev_bal * beta
            
        if positions.iloc[i] == 1:
            if beta > 1:
                leg_y[i] = prev_bal * (1/beta) + pnl_y[i]
                leg_x[i] = prev_bal - pnl_x[i]
            else:
                leg_y[i] = prev_bal + pnl_y[i]
                leg_x[i] = prev_bal * beta - pnl_x[i]
        else:
            if beta > 1:
                leg_y[i] = prev_bal * (1/beta) - pnl_y[i]
                leg_x[i] = prev_bal + pnl_x[i]
            else:
                leg_y[i] = prev_bal - pnl_y[i]
                leg_x[i] = prev_bal * beta + pnl_x[i]

    def _apply_costs(self, i, beta, positions, durations, pnl_y, pnl_x, investment, is_new=False):
        """Applies trading costs to P&L."""
        commission = 0.0028
        short_cost = 0.01/252
        
        if beta >= 1:
            pnl_y[i] -= commission * (1/beta) * investment
            pnl_x[i] -= commission * investment
            if is_new and positions.iloc[i] == 1:
                pnl_x[i] -= short_cost * investment
            elif is_new and positions.iloc[i] == -1:
                pnl_y[i] -= short_cost * (1/beta) * investment
        else:
            pnl_y[i] -= commission * investment
            pnl_x[i] -= commission * beta * investment
            if is_new and positions.iloc[i] == 1:
                pnl_x[i] -= short_cost * beta * investment
            elif is_new and positions.iloc[i] == -1:
                pnl_y[i] -= short_cost * investment

    def _apply_short_costs(self, i, beta, positions, durations, pnl_y, pnl_x, investment):
        """Applies short costs when closing positions."""
        short_cost = 0.01/252
        dur = durations.iloc[i]
        
        if positions.iloc[i] == 1:
            if beta > 1:
                pnl_x[i] -= dur * short_cost * investment
            else:
                pnl_x[i] -= dur * short_cost * beta * investment
        else:
            if beta > 1:
                pnl_y[i] -= dur * short_cost * (1/beta) * investment
            else:
                pnl_y[i] -= dur * short_cost * investment

    def _compile_balance_results(self, y, x, balance, pnl_y, pnl_x, leg_y, leg_x, triggers, positions, durations):
        """Compiles all balance results into DataFrame."""
        pnl = [pnl_y[i] + pnl_x[i] for i in range(len(y))]
        
        result = pd.DataFrame({
            'account_balance': balance,
            'returns': pd.Series(balance, index=y.index).pct_change().fillna(0),
            'pnl': pnl,
            'pnl_y': pnl_y,
            'pnl_x': pnl_x,
            'leg_y': leg_y,
            'leg_x': leg_x,
            'position_trigger': triggers,
            'positions': positions,
            'y': y,
            'x': x,
            'trading_duration': durations
        })
        
        return result

    def trade_summary(self, series, beta=0):
        """Compiles trade summary DataFrame."""
        for attr, name in series:
            try:
                attr.name = name
            except:
                continue
                
        summary = pd.concat([item[0] for item in series], axis=1)
        summary['numUnits'] = summary['numUnits'].shift().fillna(0)
        summary = summary.rename(columns={"numUnits": "position_during_day"})
        summary['position_ret_with_costs'] = self._add_transaction_costs(summary, beta)
        
        return summary

    def _add_transaction_costs(self, summary, beta=0, commission=0.08, impact=0.2, short_cost=1):
        """Adds transaction costs to returns."""
        fixed_cost = (commission + impact) / 100
        daily_short = (short_cost / 252) / 100
        
        costs = summary.apply(
            lambda row: self._apply_cost_row(row, fixed_cost, daily_short, beta), 
            axis=1
        )
        
        return summary['position_return'] - costs

    def _apply_cost_row(self, row, fixed_cost, short_cost, beta):
        """Calculates costs for a single row."""
        if beta == 0 and 'beta_position' in row:
            beta = row['beta_position']
            
        if row['position_during_day'] == 0 or row['trading_duration'] == 0:
            return 0
            
        if row['position_during_day'] == 1.:
            if beta >= 1:
                return fixed_cost*(1/beta) + fixed_cost + short_cost*row['trading_duration']
            return fixed_cost*beta + fixed_cost + short_cost*row['trading_duration']*beta
        else:
            if beta >= 1:
                return fixed_cost*(1/beta) + fixed_cost + short_cost*row['trading_duration']*(1/beta)
            return fixed_cost*beta + fixed_cost + short_cost*row['trading_duration']

    def _calculate_sharpe(self, n_years, n_days, returns):
        """Calculates annualized Sharpe ratio."""
        daily_idx = returns.resample('D').last().dropna().index
        daily_ret = (returns + 1).resample('D').prod() - 1
        daily_ret = daily_ret.loc[daily_idx]
        
        annual_ret = (np.cumprod(1 + returns) - 1).iloc[-1]
        year = returns.index[0].year
        
        if year in self.rf_rates:
            return (annual_ret - self.rf_rates[year]) / (np.std(daily_ret)*np.sqrt(n_years*n_days))
        return annual_ret / (np.std(daily_ret)*np.sqrt(n_years*n_days))

    def analyze_results(self, sharpe_results, cum_returns, performance, pairs, ticker_segments, n_years):
        """Analyzes and summarizes strategy performance."""
        avg_total, avg_annual, positive_pct = self._calculate_metrics(cum_returns, n_years)
        portfolio_sharpe = self._calculate_portfolio_sharpe(performance, pairs)
        
        # Create pairs DataFrame
        pairs_data = []
        sorted_idx = np.flip(np.argsort(sharpe_results))
        
        for idx in sorted_idx:
            pos_ret = performance[idx][1]['position_ret_with_costs']
            pos_pos = len(pos_ret[pos_ret > 0])
            neg_pos = len(pos_ret[pos_ret < 0])
            
            pairs_data.append([
                pairs[idx][0], ticker_segments[pairs[idx][0]],
                pairs[idx][1], ticker_segments[pairs[idx][1]],
                pairs[idx][2]['t_statistic'], pairs[idx][2]['p_value'],
                pos_pos, neg_pos, sharpe_results[idx]
            ])
            
        pairs_df = pd.DataFrame(
            pairs_data,
            columns=['Leg1', 'Leg1_Segmt', 'Leg2', 'Leg2_Segmt', 
                    't_statistic', 'p_value', 'positive_trades',
                    'negative_trades', 'sharpe_result']
        )
        
        pairs_df['positive_trades_pct'] = (
            pairs_df['positive_trades'] / 
            (pairs_df['positive_trades'] + pairs_df['negative_trades']) * 100
        )
        
        avg_positive_pct = pairs_df['positive_trades_pct'].mean()
        
        # Calculate drawdown
        total_balance = performance[0][1]['account_balance']
        for idx in range(1, len(pairs)):
            total_balance = total_balance + performance[idx][1]['account_balance']
        total_balance = total_balance.ffill()
        
        max_dd, dd_duration, total_dd = self._calculate_max_drawdown(total_balance)
        
        results = {
            'n_pairs': len(sharpe_results),
            'portfolio_sharpe': portfolio_sharpe,
            'avg_total_roi': avg_total,
            'avg_annual_roi': avg_annual,
            'positive_trades_pct': avg_positive_pct,
            'positive_pairs_pct': positive_pct,
            'max_drawdown': max_dd,
            'max_dd_duration': dd_duration,
            'total_dd_duration': total_dd
        }
        
        return results, pairs_df

    def _calculate_metrics(self, cum_returns, n_years):
        """Calculates performance metrics."""
        avg_total = np.mean(cum_returns)
        avg_annual = ((1 + (avg_total / 100)) ** (1 / float(n_years)) - 1) * 100
        positive_pct = len(np.array(cum_returns)[np.array(cum_returns) > 0]) * 100 / len(cum_returns)
        
        return avg_total, avg_annual, positive_pct

    def _calculate_portfolio_sharpe(self, performance, pairs):
        """Calculates portfolio-level Sharpe ratio."""
        balance = performance[0][1]['account_balance'].resample('D').last().dropna()
        returns = balance.pct_change().fillna(0)
        
        for idx in range(1, len(pairs)):
            bal = performance[idx][1]['account_balance'].resample('D').last().dropna()
            balance = balance + bal
            returns = pd.concat([returns, bal.pct_change().fillna(0)], axis=1)
            
        # Add initial balance
        init_bal = pd.Series([len(pairs)], index=[balance.index[0] - timedelta(days=1)])
        balance = pd.concat([init_bal, balance])
        
        # Calculate volatility
        weights = np.array([1/len(pairs)] * len(pairs))
        vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        
        # Calculate return
        annual_ret = (balance.iloc[-1] - len(pairs))/len(pairs)
        year = balance.index[-1].year
        
        if year in self.rf_rates:
            return (annual_ret - self.rf_rates[year]) / (vol*np.sqrt(252))
        return annual_ret / (vol*np.sqrt(252))

    def _calculate_max_drawdown(self, balance):
        """Calculates maximum drawdown statistics."""
        # Total drawdown days
        dd_days = balance.resample('D').last().dropna().diff().fillna(0).apply(
            lambda x: 0 if x >= 0 else 1).sum()
        
        # Max drawdown
        values = np.asarray(balance.values)
        peak_idx = np.argmax(np.maximum.accumulate(values) - values)
        
        if peak_idx == 0:
            return 0, 0, dd_days
            
        trough_idx = np.argmax(values[:peak_idx])
        dd_period = round((peak_idx - trough_idx))  # Approximation
        
        # Plot
        plt.figure(figsize=(10,7))
        plt.grid()
        plt.plot(values, label='Account Balance')
        dates = balance.resample('BMS').first().dropna().index.date
        xi = np.linspace(0, len(balance), len(dates))
        plt.xticks(xi, dates, rotation=50)
        plt.xlim(0, len(balance))
        plt.plot([peak_idx, trough_idx], [values[peak_idx], values[trough_idx]], 
                'o', color='Red', markersize=10)
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        
        return (values[peak_idx]-values[trough_idx])/values[trough_idx]*100, dd_period, dd_days

    def construct_portfolio(self, performance_data: List, 
                          weights: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Construct weighted portfolio from pairs trading results
        
        Args:
            performance_data: List of performance tuples from strategy
            weights: Dictionary of weights for each pair (e.g., {'EMR_PH': 0.2})
                    If None, equal weights will be used
        """
        if not performance_data:
            raise ValueError("No performance data provided")
            
        # Process weights
        weights = self._process_weights(weights, performance_data)
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(performance_data, weights)
        
        # Calculate statistics
        stats = self._calculate_portfolio_stats(portfolio)
        
        return portfolio, stats
    
    def _process_weights(self, weights: Optional[Dict], 
                        performance_data: List) -> Dict:
        """Process and normalize weights"""
        if weights is None:
            pair_names = [f"{p[0][0]}_{p[0][1]}" for p in performance_data if p is not None]
            return {name: 1/len(pair_names) for name in pair_names}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}
    
    def _initialize_portfolio(self, performance_data: List, 
                            weights: Dict) -> pd.DataFrame:
        """Initialize portfolio with weighted returns"""
        portfolio = pd.DataFrame()
        
        for (pair, summary, _) in performance_data:
            if pair is None or summary is None:
                continue
                
            pair_name = f"{pair[0]}_{pair[1]}"
            weight = weights.get(pair_name, 0)
            
            # Calculate weighted returns
            weighted_returns = summary['returns'] * weight
            weighted_balance = summary['account_balance'] * weight * self.initial_capital
            
            if portfolio.empty:
                portfolio = pd.DataFrame({
                    'date': summary.index,
                    'total': weighted_balance,
                    pair_name: weighted_balance
                }).set_index('date')
            else:
                portfolio['total'] = portfolio['total'].add(weighted_balance, fill_value=0)
                portfolio[pair_name] = weighted_balance.reindex(portfolio.index, method='ffill').fillna(0)
        
        portfolio['daily_returns'] = portfolio['total'].pct_change().fillna(0)
        portfolio['cumulative_returns'] = (1 + portfolio['daily_returns']).cumprod() - 1
        
        return portfolio
    
    def _calculate_portfolio_stats(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate portfolio performance statistics"""
        if portfolio.empty:
            return {}
            
        daily_returns = portfolio['daily_returns']
        total_return = portfolio['total'].iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252/len(portfolio)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        running_max = portfolio['total'].cummax()
        drawdown = (running_max - portfolio['total']) / running_max
        max_drawdown = drawdown.max()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'drawdown': drawdown
        }

    def summarize_results(self, sharpe_results, cum_returns, performance, total_pairs, 
                         ticker_segment_dict, n_years, weights=None):
        """
        Summarize strategy results with optional weighted portfolio
        
        Args:
            weights: Dictionary of pair weights (e.g., {'EMR_PH': 0.2})
        """
        # Analyze results to get metrics and pairs_df
        metrics, pairs_df = self.analyze_results(
            sharpe_results, cum_returns, performance, 
            total_pairs, ticker_segment_dict, n_years
        )
        
        # Construct portfolio
        portfolio, portfolio_stats = self.construct_portfolio(performance, weights)
        
        # Update metrics with portfolio stats
        metrics.update({
            'portfolio_total_return': portfolio_stats['total_return'],
            'portfolio_annualized_return': portfolio_stats['annualized_return'],
            'portfolio_volatility': portfolio_stats['volatility'],
            'portfolio_sharpe_ratio': portfolio_stats['sharpe_ratio'],
            'portfolio_max_drawdown': portfolio_stats['max_drawdown']
        })
        
        return metrics, pairs_df, portfolio
