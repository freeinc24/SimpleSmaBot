from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import pandas as pd
import ta
import asyncio
from datetime import timedelta
import logging

class PocketOptionBot:
    def __init__(self):
        self.amount = 1
        self.initial_amount = 1
        self.symbol = "EURUSD_otc"
        self.martingale_steps = 4
        self.current_step = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Strategy parameters - optimized for better signals
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_smooth_k = 3
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.min_candles = 30  # Increased for more reliable indicators
        
        # Risk management
        self.max_consecutive_losses = 5
        self.cooldown_period = 3  # Skip trades after max losses
        self.cooldown_counter = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_stochastic_signals(self, df):
        """Calculate Stochastic Oscillator signals"""
        try:
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'], 
                low=df['low'], 
                close=df['close'],
                window=self.stoch_k_period,
                smooth_window=self.stoch_smooth_k
            )
            
            k_line = stoch.stoch()
            d_line = stoch.stoch_signal()
            
            # Current and previous values for crossover detection
            k_current = k_line.iloc[-1]
            k_previous = k_line.iloc[-2]
            d_current = d_line.iloc[-1]
            d_previous = d_line.iloc[-2]
            
            # Detect crossovers
            k_crosses_d_up = (k_previous <= d_previous) and (k_current > d_current)
            k_crosses_d_down = (k_previous >= d_previous) and (k_current < d_current)
            
            return {
                'k_crosses_d_up': k_crosses_d_up,
                'k_crosses_d_down': k_crosses_d_down,
                'k_current': k_current,
                'd_current': d_current,
                'oversold': k_current < 20,
                'overbought': k_current > 80
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return None

    def calculate_macd_signals(self, df):
        """Calculate MACD signals"""
        try:
            macd_indicator = ta.trend.MACD(
                close=df['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            
            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()
            
            # Current and previous values
            macd_current = macd_line.iloc[-1]
            signal_current = signal_line.iloc[-1]
            histogram_current = histogram.iloc[-1]
            histogram_previous = histogram.iloc[-2]
            
            return {
                'macd_below_signal': macd_current < signal_current,
                'macd_above_signal': macd_current > signal_current,
                'macd_current': macd_current,
                'signal_current': signal_current,
                'histogram_current': histogram_current,
                'histogram_increasing': histogram_current > histogram_previous,
                'histogram_decreasing': histogram_current < histogram_previous
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return None

    def generate_trading_signal(self, df):
        """Generate trading signal based on Stochastic + MACD strategy"""
        stoch_data = self.calculate_stochastic_signals(df)
        macd_data = self.calculate_macd_signals(df)
        
        if not stoch_data or not macd_data:
            return "HOLD", "Indicator calculation failed"
        
        # CALL signal conditions:
        # 1. Stochastic K line crosses D line going up
        # 2. MACD line is below signal line (but showing potential for upward momentum)
        call_condition = (
            stoch_data['k_crosses_d_up'] and 
            macd_data['macd_below_signal'] and
            macd_data['histogram_increasing']  # Additional confirmation
        )
        
        # PUT signal conditions:
        # 1. Stochastic K line crosses D line going down
        # 2. MACD line is above signal line (but showing potential for downward momentum)
        put_condition = (
            stoch_data['k_crosses_d_down'] and 
            macd_data['macd_above_signal'] and
            macd_data['histogram_decreasing']  # Additional confirmation
        )
        
        # Generate signal with reasoning
        if call_condition:
            reason = f"CALL: Stoch K crossed D up ({stoch_data['k_current']:.2f}>{stoch_data['d_current']:.2f}), MACD below signal with increasing momentum"
            return "CALL", reason
        elif put_condition:
            reason = f"PUT: Stoch K crossed D down ({stoch_data['k_current']:.2f}<{stoch_data['d_current']:.2f}), MACD above signal with decreasing momentum"
            return "PUT", reason
        else:
            return "HOLD", "No clear signal detected"

    def update_martingale(self, result):
        """Update martingale progression with safety limits"""
        if result == 'win':
            self.amount = self.initial_amount
            self.current_step = 0
            self.consecutive_losses = 0
            self.winning_trades += 1
            self.cooldown_counter = 0  # Reset cooldown on win
        elif result == 'loss':
            self.consecutive_losses += 1
            if self.current_step < self.martingale_steps - 1:
                self.current_step += 1
                self.amount = self.initial_amount * (2 ** self.current_step)
                self.logger.info(f"Martingale step {self.current_step}, new amount: {self.amount}")
            else:
                # Reset after max martingale steps
                self.logger.warning("Max martingale steps reached, resetting")
                self.amount = self.initial_amount
                self.current_step = 0
                self.cooldown_counter = self.cooldown_period
        
        self.total_trades += 1
        
        # Implement cooldown after too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.cooldown_counter = self.cooldown_period
            self.consecutive_losses = 0
            self.logger.warning(f"Too many consecutive losses, entering cooldown for {self.cooldown_period} trades")

    def should_trade(self):
        """Determine if we should trade based on risk management rules"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        return True

    def log_performance(self):
        """Log current performance statistics"""
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            self.logger.info(f"Performance: {self.winning_trades}/{self.total_trades} wins ({win_rate:.1f}%), Current amount: {self.amount}")

    async def execute_trade(self, client, signal, reason):
        """Execute trade with error handling"""
        try:
            self.logger.info(f"Executing {signal} trade: {reason}")
            
            if signal == "CALL":
                buy_id, _ = await client.buy(
                    asset=self.symbol, 
                    amount=self.amount, 
                    time=5, 
                    check_win=False
                )
            else:  # PUT
                buy_id, _ = await client.sell(
                    asset=self.symbol, 
                    amount=self.amount, 
                    time=5, 
                    check_win=False
                )
            
            # Wait for trade result
            buy_data = await client.check_win(buy_id)
            result = buy_data['result']
            
            self.logger.info(f"Trade result: {result}")
            self.update_martingale(result)
            self.log_performance()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    async def main_bot(self, ssid):
        """Main bot loop with enhanced error handling and strategy"""
        try:
            self.logger.info("Starting PocketOption Bot with Stochastic + MACD strategy")
            client = PocketOptionAsync(ssid=ssid)
            await asyncio.sleep(5)

            stream = await client.subscribe_symbol_timed(
                self.symbol, 
                timedelta(seconds=2)
            )

            candles_list = []
            
            self.logger.info(f"Bot started. Collecting initial {self.min_candles} candles...")
            
            async for candle in stream:
                candles_list.append(candle)
                
                # Need minimum candles for reliable indicator calculation
                if len(candles_list) < self.min_candles:
                    if len(candles_list) % 10 == 0:  # Progress update every 10 candles
                        self.logger.info(f"Collected {len(candles_list)}/{self.min_candles} candles")
                    continue
                
                # Maintain rolling window of candles
                if len(candles_list) > 100:
                    candles_list = candles_list[-100:]
                
                try:
                    # Create DataFrame and generate signal
                    candles_df = pd.DataFrame(candles_list)
                    signal, reason = self.generate_trading_signal(candles_df)
                    
                    self.logger.info(f"Signal: {signal} - {reason}")
                    
                    # Execute trade if conditions are met
                    if signal in ["CALL", "PUT"] and self.should_trade():
                        await self.execute_trade(client, signal, reason)
                    elif not self.should_trade():
                        self.logger.info("Trade skipped due to cooldown period")
                    
                except Exception as e:
                    self.logger.error(f"Error processing candle: {e}")
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Critical error in main bot: {e}")
            raise

if __name__ == '__main__':
    print("Enhanced PocketOption Bot - Stochastic + MACD Strategy (30s/30s)")
    print("=" * 60)
    
    ssid = input('Please enter your SSID: ')
    
    bot = PocketOptionBot()
    
    print(f"\nBot Configuration:")
    print(f"Symbol: {bot.symbol}")
    print(f"Candle Timeframe: 30 seconds")
    print(f"Trade Expiry: 30 seconds")
    print(f"Initial Amount: {bot.initial_amount}")
    print(f"Martingale Steps: {bot.martingale_steps}")
    print(f"Stochastic Period: {bot.stoch_k_period}")
    print(f"MACD: {bot.macd_fast}/{bot.macd_slow}/{bot.macd_signal}")
    print(f"Min Candles: {bot.min_candles}")
    print("=" * 60)
    
    try:
        asyncio.run(bot.main_bot(ssid))
    except Exception as e:
        print(f"Bot crashed: {e}")
        input("Press Enter to exit...")
