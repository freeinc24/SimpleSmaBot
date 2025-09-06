from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from datetime import timedelta
import time
import asyncio

# Main part of the code
async def main(ssid: str):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)    
    time.sleep(5)
    stream = await api.subscribe_symbol_timed("EURUSD_otc", timedelta(seconds=5)) # Returns a candle obtained from combining candles that are inside a specific time range
    
    # This should run forever so you will need to force close the program
    async for candle in stream:
        print(f"{candle}") # Each candle is in format of a dictionary 
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
    