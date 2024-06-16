# Trader Bot



Trader Bot is my first attempt to create a bot using [Alpaca](https://alpaca.markets) API It allows for collection of all the stock symbols and trying to find "Strong Buys"



Keep in mind I know nothing about stock trading and I am only testing this in the paper enviroment.



It's a CLI application

![](https://github.com/john8675309/bot_trader/blob/main/help.png?raw=true)



**analyze <ticker>**
See if this could be a good buy

![](https://github.com/john8675309/bot_trader/blob/main/analyze.png?raw=true)



**exit**

quit



**marketstatus**

is the market open or not

![](https://github.com/john8675309/bot_trader/blob/main/marketstatus.png?raw=true)



**showorders**

show open orders

![](https://github.com/john8675309/bot_trader/blob/main/showorders.png?raw=true)



**buy <ticker> <shares> <market (optional default Limit)>**

but ticker number of shares

![](https://github.com/john8675309/bot_trader/blob/main/buy.png?raw=true)



**getbalance**

Alpaca Balance

**be cautious of margin**

![](https://github.com/john8675309/bot_trader/blob/main/balance.png?raw=true)



**monitor**

Automatically started not needed. Monitor thread to to monitor Orders to filled



**showpositions**

This is untested (markets closed, when I wrote it). should show what you own



**createlimitorders**

After you run:

downloadsymbols

readtickers db db

you can run createlimitorders this will drain your balance and buy "Strong Buy" stocks

**This is the "magic"**



**gettime**

Time from Alpaca

![](https://github.com/john8675309/bot_trader/blob/main/gettime.png?raw=true)



**readtickers <in_file> <out_file>**

you will mainly want to use:

downloadsymbols

readtickers db db

but you can do

readtickers <in_file> <out_file, or db>

this will read tickers from a csv and output them to either an out_file csv or db with "No Buy", "Buy", or "Strong Buy"

readtickers db db reads the tickers from the DB and writes them out to the DB.



**showrecommendations <type>**

takes argument of "Buy", "No Buy", "Strong Buy", or none for all, will show based on our algorithm what to do.

I.E. showreommendations Strong Buy

![](https://github.com/john8675309/bot_trader/blob/main/showrecommendations.png?raw=true)



**downloadsymbols**

Download and store all the symbols from the SEC



**help**

Show help

![](https://github.com/john8675309/bot_trader/blob/main/help.png?raw=true)



**sell ticker quantity <market (optional default Limit)]>**

sell stock you own

Untested (Markets Closed)














