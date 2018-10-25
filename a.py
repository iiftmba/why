
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.  Load and Parse spreadsheet
Exl = pd.ExcelFile('C:/Users/anjali/Desktop/BNYMellon_Anjali/stockPrice.xlsx')
Exl.sheet_names

df1 = Exl.parse('price')
StPrices = df1.sort_values(by='Date')
StPrices.set_index('Date', inplace=True)
print(StPrices.head())

df2 = Exl.parse('mktCap')    
MarketCap = df2.sort_values(by='Date')
MarketCap.set_index('Date', inplace=True)
print(MarketCap.head())

###############################################################################
# 2a. For each of these stock markets, Calculate the rolling one-year risk and 
#     rolling one-year correlations. 
#     Plot time series for both 
# As question Note : Risk is standard deviation, hence I have taken standard deviation of daily return
###############################################################################
# CODE TO CALUCULATE RETURN OF EACH STOCK MARKET


Stock_Return_List = ['AusStk_ret', 'CanStk_ret', 'FraStk_ret', 'DeuStk_ret', 'HkgStk_ret', 'ItaStk_ret', 'JpnStk_ret', 'NldStk_ret', 'EspStk_ret', 'CheStk_ret', 'GbrStk_ret', 'UsaStk_ret']
Stock_Return_List

Stock_List = StPrices.columns[0:12]
daily_return = pd.DataFrame()
for x in Stock_List:   
    daily_return[x] = StPrices[x].pct_change() 
print(daily_return.tail())   

# plotting Risk and Correl in single graph

plt.plot(daily_return)
plt.legend()
plt.title('Graph for Daily Return')
plt.xlabel('Period')
plt.ylabel('Return')
plt.show() 

# Calculate the rolling one-year risk for each of stock market
year_rolling_risk = pd.DataFrame()
for i in daily_return:
    year_rolling_risk[i] = daily_return[i].rolling(window=260,center=False).std()
print(year_rolling_risk.tail())

# Calculate the rolling one-year correlations for each of stock market with each other

year_rolling_correl = pd.DataFrame()
for j in daily_return:
    for k in daily_return:
        year_rolling_correl[j] = daily_return[j].rolling(window=260).corr(other = daily_return[k])          
print(year_rolling_correl.tail())

#-------------------------------------------------------------
# Time Series Plot for rolling one-year risk and correlations 
plt.plot(year_rolling_risk["AusStk"])
plt.legend()
plt.title('Time Series Graph for Rolling Risk')
plt.xlabel('Period')
plt.ylabel('Rolling Risk')
plt.show() 


# CODE TO PLOT TIME SERIES OF ROLLING RISK FOR ALL MARKET USING MATPLOTLIB
plt.plot(year_rolling_risk)
plt.legend()
plt.title('Time Series Graph for Rolling Risk')
plt.xlabel('Period')
plt.ylabel('Rolling Risk')
plt.show()

# CODE TO PLOT ROLLING YEARLY CORRELATION BETWEEN RETURN OF TWO MARKETS

# Correlation between return of AusStk Market to CanStk Market             
plt.plot(year_rolling_correl["AusStk"])
plt.legend()
plt.title('Time Series Graph for Rolling Correlation')
plt.xlabel('Period')
plt.ylabel('Rolling Correlation')
plt.show()

# CODE TO PLOT TIME SERIES OF ROLLING RISK FOR ALL MARKET USING MATPLOTLIB
plt.plot(year_rolling_correl)
plt.legend()
plt.title('Time Series Graph for Rolling Correlation')
plt.xlabel('Period')
plt.ylabel('Rolling Correlation')
plt.show()
print()

# plotting Risk and Correl in single graph
plt.plot(year_rolling_risk["AusStk"])
plt.plot(year_rolling_correl["AusStk"])
plt.legend()
plt.title('Graph for Rolling Risk vs. Rolling Correl')
plt.xlabel('Period')
plt.ylabel('Value')
plt.show() 

###############################################################################
#  2b. Calculate the average annualized return, risk, return-risk ratio and 
#      the maximum drawdown for the whole period
###############################################################################

# CODE TO CALCULATE OVERALL RETURN

mean_return = []
for p in daily_return[1:1]:
    # To calculate the average daily return, we use the np.mean() function:
    mret = np.mean(daily_return[p])
    print("StockMarket : %s , Mean Return : %2.6f" %(p, mret))
    mean_return.append(mret)
      
best_return = max(mean_return)
best_index = mean_return.index(best_return)
best_market = Stock_List[best_index] 
print(best_market,"is best Market with Highest Mean Return : %2.6f" %(best_return))

# CODE TO CALCULATE AVERAGE ANNUALISED RETURN

avg_annualised_return = []
for q in daily_return[1:1]:
    avgreturn = ((1+np.mean(daily_return[q]))**260)-1
    print("StockMarket : %s , Average Annualised Return : %2.6f" %(q, avgreturn))
    avg_annualised_return.append(avgreturn)

best_annual_return = max(avg_annualised_return)
best_annual_index = avg_annualised_return.index(best_annual_return) 
best_annual_market = Stock_List[best_annual_index] 
print(best_annual_market,"is best Market with Highest Average Annualised Return : %2.6f" %(best_annual_return))


# CODE TO CALCULATE OVERALL PERIOD RISK
mean_risk = []
for r in daily_return[1:1]:
    mrisk = np.std(daily_return[r])
    print("StockMarket : %s , Overall Risk : %2.6f" %(r, mrisk))
    mean_risk.append(mrisk)
      
highest_risk = max(mean_risk)
h_index = mean_risk.index(highest_risk) 
riskiest_market = Stock_List[h_index] 
print(riskiest_market,"is the Riskiest Market with Highest Overall Risk : %2.6f" %(highest_risk))

# CODE TO CALCULATE AVERAGE ANNUALIZED RISK :
annualised_risk = []
for s in daily_return[1:1]:
    annualrisk = np.std(daily_return[s]) * np.sqrt(260)
    print("StockMarket : %s , Average Annualised Risk : %2.6f" %(s, annualrisk))
    annualised_risk.append(annualrisk)

max_annul_avgrisk = max(annualised_risk)
max_index = annualised_risk.index(max_annul_avgrisk) 
annualised_riskiest_market = Stock_List[max_index] 
print(annualised_riskiest_market,"is the Riskiest Market with Highest Average Annualised Risk : %2.6f" %(max_annul_avgrisk))

# CODE FOR AVERAGE ANNUALISED RETURN-RISK RATIO

return_risk_ratio = []
c = 1
for t in range(12):
    rr_ratio = avg_annualised_return[t]/annualised_risk[t]
    print("StockMarket : %2d , Avg Annual Return Risk Ratio : %2.6f" %(c, rr_ratio))
    c = c+1
    return_risk_ratio.append(rr_ratio)
    
# CODE FOR SHARPE RATIO

rf = 0.005  # the risk-free rate (rf), We can change it as per market

sharpe_ratio = []
c = 1
for t in range(12):
    s_ratio = (avg_annualised_return[t] - rf) / annualised_risk[t]
    print("StockMarket : %2d , Sharpe Ratio : %2.6f" %(c, s_ratio))
    c = c+1
    sharpe_ratio.append(s_ratio)
    
# CODE TO DECIDE WHICH IS BEST MARKET AS PER AVERAGE ANNUALISED RETURN

highest_rratio = max(return_risk_ratio)
highest_sratio = max(sharpe_ratio)

rr_index = return_risk_ratio.index(highest_rratio) 
sharpe_index = sharpe_ratio.index(highest_sratio) 

best_rr_market = Stock_List[rr_index] 
best_sharpe_market = Stock_List[sharpe_index]

print(best_rr_market,"is best Market with Highest Average Annualised Return Risk Ratio : %2.6f" %(highest_rratio))
print(best_sharpe_market,"is best Market with Highest Sharpe Ratio : %2.6f" %(highest_sratio))

# Code for the maximum drawdown for the whole period

def maxdrawdown(equity_price):
    # Code to pull end of the drawdown period
    i = np.argmax(np.maximum.accumulate(equity_price.values) - equity_price.values) 
    # Code to pull start of drawdown period
    j = np.argmax(equity_price.values[:i]) 
    # I have done absolute value of drawdown to make it positive from negative
    max_drawdown = abs(100.0*(equity_price[i]-equity_price[j])/equity_price[j])

    equity_value = equity_price.index.values
    start_dt = pd.to_datetime(str(equity_value[j]))
    drawdown_start=start_dt.strftime ("%Y-%m-%d") 

    end_dt = pd.to_datetime(str(equity_value[i]))
    drawdown_end = end_dt.strftime ("%Y-%m-%d") 

    drawdown_duration=np.busday_count(drawdown_start, drawdown_end)

    return drawdown_start, drawdown_end, drawdown_duration, max_drawdown, 

# CODE TO PRINT MAX DRAWDOWN FOR ALL STOCK FOR WHOLE PERIOD 
for v in Stock_List:   
    d = maxdrawdown(StPrices[v])
    print("Market: ",v,", Start Date: ", d[0], ", End Date: ", d[1])
    print("Duration in days: ",d[2], ", Max Drawdown Percent: %3.2f" %(d[3]))
    print()

# 2c. Provide the analysis regarding which stock market is more attractive and why.
    
print(best_market,"is best Market with Highest Mean Return : %2.6f" %(best_return))
print(best_annual_market,"is best Market with Highest Average Annualised Return : %2.6f" %(best_annual_return))
print(riskiest_market,"is the Riskiest Market with Highest Overall Risk : %2.6f" %(highest_risk))
print(annualised_riskiest_market,"is the Riskiest Market with Highest Average Annualised Risk : %2.6f" %(max_annul_avgrisk))
print(best_rr_market,"is best Market with Highest Average Annualised Return Risk Ratio : %2.6f" %(highest_rratio))
print(best_sharpe_market,"is best Market with Highest Sharpe Ratio : %2.6f" %(highest_sratio))

##################################################################################
# 3. Construct a cap weighted global equity portfolio and calculate the average 
# annualized return, risk, return-risk ratio and the maximum drawdown for the 
# whole period as well as the starting and ending dates of the maximum drawdown.
##################################################################################

# Market Cap Dataframe :
Mkt_List = MarketCap.columns[0:12]

# Total Sum of Market capitalisation in the begining
tsum = 0
initial_cap = []
for i in Mkt_List: 
    cap = MarketCap[:1][i][0]
#    cap = MarketCap[:1][i]
    print("Stock : ", i , ", Market Cap : ", cap)
    initial_cap.append(cap)
    tsum = tsum + cap


# CODE TO CALCULATE WEIGHT OF CAP OF EACH MARKET : CAP/TOTAL CAP

portfolio_weight = []
portsum = 0
for j in Mkt_List: 
    port_weight = MarketCap[j][0]/tsum
    portfolio_weight.append(port_weight)
    print("Stock : ", j,", Weight : ", port_weight )
    portsum = portsum + port_weight

print("Sum of total weight :", portsum)

# CODE TO CALCUALTE WEIGHTAGE OF INVESTMENT IN EACH MARKET
# APPROX INVESTMENT ALLOCATION (CAP WEIGHTED) TO EACH STOCK

investment = 100000000   # Let we plan to invest approx 100 million in total portfolio 
print("Approximate total investment $ :", investment)
print("Part of investment $ allocated to each of Stock Market : ")

portfolio_amount = []
c = 0
for k in Mkt_List: 
    port_amount = portfolio_weight[c]*investment
    print("Stock : ", k,", Investment Acllocation $: %.2f" %(port_amount) )
    portfolio_amount.append(port_amount)
    c = c + 1

# CODE TO FIND OUT HOW MANY EACH OF STOCK TO BE BOUGHT ON INVESTMENT WEIGHT
print("NUMBER OF STOCKS (CAP WEIGHTED) OF EACH MARKET IN CLOBAL PORTFOLIO      ")

stock_count = []
c = 0
for l in Mkt_List: 
    st_count = round(portfolio_amount[c]/StPrices[:1][l][0])
    print("Stock : ", l,", No. of Stock in Portfolio : ", st_count)
    stock_count.append(st_count)
    c = c + 1

print("   ACTUAL INVESTMENT ALLOCATION (CAP WEIGHTED) TO EACH STOCK            ")

actual_invest = []
c = 0
isum = 0
for l in Mkt_List: 
    act_invest = stock_count[c]*StPrices[:1][l][0]
    print("Stock : ", l,", Actual investment $ : %.1f " %(act_invest))
    isum = isum + act_invest
    actual_invest.append(act_invest)
    c = c + 1

print("Total actual investment to create cap wt Global Portfolio : $ %.2f" %(isum)) 

# CREATION OF GLOBAL PORTFOLIO DATA FRAME WITH CAP WT
print("     GLOBAL PORTFOLIO (CAP WEIGHTED) OF EACH STOCK MARKET               ")

Global_Portfolio = pd.DataFrame()
c = 0
for x in Stock_List:   
    Global_Portfolio[x] = StPrices[x] * stock_count[c] 
    c = c+1   
print(Global_Portfolio.tail())  

print("             TIME SERIES PLOTS - GLOBAL MARKET PORFOLIO                 ")

plt.plot(Global_Portfolio)
plt.legend()
plt.title('Graph for Global Portfolio Investment')
plt.xlabel('Period')
plt.ylabel('Investment')
plt.show() 

# CODE TO CALUCULATE RETURN OF EACH STOCK MARKET


# Calculate the daily return for each of stock market and append to 'price' dataframe

print("                 GLOBAL MARKET PORTFOLIO - DAILY RETURN                 ")
print("Showing only tail data for output clarity")

Stock_List = Global_Portfolio.columns[0:12]

port_daily_return = pd.DataFrame()
for x in Stock_List:   
    port_daily_return[x] = Global_Portfolio[x].pct_change() 
    
print(port_daily_return.tail())   

print("       TIME SERIES PLOTS - GLOBAL PORTFOLIO DAILY RETURN                ")

# plotting Risk and Correl in single graph
plt.plot(port_daily_return)
plt.legend()
plt.title('Graph for Global Portfolio Daily Return')
plt.xlabel('Period')
plt.ylabel('Return')
plt.show() 

#------------------------------------------------------------
# Calculate the rolling one-year risk for each of stock market
#------------------------------------------------------------

port_year_rolling_risk = pd.DataFrame()
for i in port_daily_return:
    port_year_rolling_risk[i] = port_daily_return[i].rolling(window=260,center=False).std()
    

print("         GLOBAL PORTFOLIO ONE YEAR ROLLING RISK                         ")

print(port_year_rolling_risk.tail())

# Calculate the rolling one-year correlations for each of stock market with each other

port_year_rolling_correl = pd.DataFrame()
for j in port_daily_return:
    for k in port_daily_return:
        port_year_rolling_correl[j] = port_daily_return[j].rolling(window=260).corr(other = port_daily_return[k])  

print("              PORTFOLIO ONE YEAR ROLLING CORRELATION                    ")

print(port_year_rolling_correl.tail())

# Time Series Plot for rolling one-year risk and correlations 

print("          TIME SERIES PLOTS - YEAR ROLLING RISK SINGLE STOCK            ")

# Rolling Risk of AusStk Stock Market               
# CODE TO PLOT TIME SERIES OF ROLLING RISK SINGLE MARKET USING MATPLOTLIB
plt.plot(port_year_rolling_risk["AusStk"])
plt.legend()
plt.title('Time Series Graph for Global Portfolio Rolling Risk')
plt.xlabel('Period')
plt.ylabel('Rolling Risk')
plt.show() 
print()

# CODE TO PLOT TIME SERIES OF ROLLING RISK FOR ALL MARKET USING MATPLOTLIB
plt.plot(port_year_rolling_risk)
plt.legend()
plt.title('Time Series Graph for Rolling Risk')
plt.xlabel('Period')
plt.ylabel('Rolling Risk')
plt.show()


print("     TIME SERIES PLOTS - YEAR ROLLING CORRELATION - SINGLE MARKETS      ")

# Correlation between return of AusStk Market to CanStk Market             
plt.plot(port_year_rolling_correl["AusStk"])
plt.legend()
plt.title('Time Series Graph for Global Portfolio Rolling Correlation')
plt.xlabel('Period')
plt.ylabel('Rolling Correlation')
plt.show() 

print("TIME SERIES PLOTS - PORTFOLIO YEAR ROLLING CORRELATION - ALL MARKETS    ")

# CODE TO PLOT TIME SERIES OF ROLLING RISK FOR ALL MARKET USING MATPLOTLIB
plt.plot(port_year_rolling_correl)
plt.legend()
plt.title('Time Series Graph for Portfolio Rolling Correlation')
plt.xlabel('Period')
plt.ylabel('Rolling Correlation')
plt.show()
print()


print("  TIME SERIES PLOTS - PORTFOLIO YEAR ROLLING RISK vs. CORRELATION       ")

# plotting Risk and Correl in single graph
plt.plot(port_year_rolling_risk["AusStk"])
plt.plot(port_year_rolling_correl["AusStk"])
plt.legend()
plt.title('Graph for Portfolio Rolling Risk vs. Rolling Correl')
plt.xlabel('Period')
plt.ylabel('Value')
plt.show() 

# CODE TO CALCULATE OVERALL RETURN

print("          GLOBAL PORTFOLIO OVERALL RETURN                               ")

port_mean_return = []
for p in port_daily_return[1:1]:
    # To calculate the average daily return, we use the np.mean() function:
    pmret = np.mean(port_daily_return[p])
    print("StockMarket : %s , Portfolio Mean Return : %2.6f" %(p, pmret))
    port_mean_return.append(pmret)
      

# CODE TO DECIDE WHICH IS BEST MARKET AS PER HIGHEST MEAN RETURN

port_best_return = max(port_mean_return)
port_best_index = port_mean_return.index(port_best_return) 
port_best_market = Stock_List[port_best_index] 
print()
print(port_best_market,"is best Market with Highest Mean Return : %2.6f" %(port_best_return))


# CODE TO CALCULATE AVERAGE ANNUALISED RETURN

port_avg_annualised_return = []
for q in port_daily_return[1:1]:
    port_avgreturn = ((1+np.mean(port_daily_return[q]))**260)-1
    print("StockMarket : %s , Portfolio Average Annualised Return : %2.6f" %(q, port_avgreturn))
    port_avg_annualised_return.append(port_avgreturn)
    
# CODE TO DECIDE WHICH IS BEST MARKET AS PER AVERAGE ANNUALISED RETURN

port_best_annual_return = max(port_avg_annualised_return)
port_best_annual_index = port_avg_annualised_return.index(port_best_annual_return) 
port_best_annual_market = Stock_List[port_best_annual_index] 

print(port_best_annual_market,"is best Market with Highest Average Annualised Return : %2.6f" %(port_best_annual_return))

# CODE TO CALCULATE OVERALL PERIOD RISK

print("              GLOBAL PORTFOLIO OVERALL PERIOD RISK                      ")

port_mean_risk = []
for r in port_daily_return[1:1]:
    pmrisk = np.std(port_daily_return[r])
    print("StockMarket : %s , Overall Risk : %2.6f" %(r, pmrisk))
    port_mean_risk.append(pmrisk)
      

# CODE TO DECIDE WHICH IS THE RISKIEST MARKET AS PER HIGHEST OVERALL RISK

port_highest_risk = max(port_mean_risk)
port_h_index = port_mean_risk.index(port_highest_risk) 
port_riskiest_market = Stock_List[port_h_index] 

print(port_riskiest_market,"is the Riskiest Market with Highest Overall Risk : %2.6f" %(port_highest_risk))

# CODE TO CALCULATE AVERAGE ANNUALIZED RISK :

print("      GLOBAL PORTFOLIO AVERAGE ANNUALIZED RISK           ")

port_annualised_risk = []
for s in port_daily_return[1:1]:
    port_annualrisk = np.std(port_daily_return[s]) * np.sqrt(260)
    print("StockMarket : %s , Average Annualised Risk : %2.6f" %(s, port_annualrisk))
    port_annualised_risk.append(port_annualrisk)

# CODE TO DECIDE WHICH IS THE RISKIEST MARKET AS PER HIGHEST OVERALL RISK

port_max_annul_avgrisk = max(port_annualised_risk)
port_max_index = port_annualised_risk.index(port_max_annul_avgrisk) 
port_annualised_riskiest_market = Stock_List[port_max_index] 

print(port_annualised_riskiest_market,"is the Riskiest Market with Highest Average Annualised Risk : %2.6f" %(port_max_annul_avgrisk))

# CODE FOR AVERAGE ANNUALISED RETURN-RISK RATIO

print("         GLOBAL PORTFOLIO AVERAGE ANNUALISED RETURN-RISK                ")

port_return_risk_ratio = []
c = 1
for t in range(12):
    port_rr_ratio = port_avg_annualised_return[t]/port_annualised_risk[t]
    print("StockMarket : %2d , Avg Annual Return Risk Ratio : %2.6f" %(c, port_rr_ratio))
    c = c+1
    port_return_risk_ratio.append(port_rr_ratio)
    
# CODE FOR SHARPE RATIO

rf = 0.005  # the risk-free rate (rf), We can change it as per market


port_sharpe_ratio = []
c = 1
for t in range(12):
    port_s_ratio = (port_avg_annualised_return[t] - rf) / port_annualised_risk[t]
    print("StockMarket : %2d , Sharpe Ratio : %2.6f" %(c, port_s_ratio))
    c = c+1
    port_sharpe_ratio.append(port_s_ratio)
    
# CODE TO DECIDE WHICH IS BEST MARKET AS PER AVERAGE ANNUALISED RETURN

port_highest_rratio = max(port_return_risk_ratio)
port_highest_sratio = max(port_sharpe_ratio)

port_rr_index = port_return_risk_ratio.index(port_highest_rratio) 
port_sharpe_index = port_sharpe_ratio.index(port_highest_sratio) 

port_best_rr_market = Stock_List[port_rr_index] 
port_best_sharpe_market = Stock_List[port_sharpe_index]

print()
print(port_best_rr_market,"is best Market with Highest Average Annualised Return Risk Ratio : %2.6f" %(port_highest_rratio))
print(port_best_sharpe_market,"is best Market with Highest Sharpe Ratio : %2.6f" %(port_highest_sratio))

# Code for the maximum drawdown for the whole period
print("          GLOBAL PORTFOLIO MAXIMUM DRAWDOWN FOR WHOLE PERIOD            ")


def portfoliomaxdrawdown(port_equity_price):
    # Code to pull end of the drawdown period
    i = np.argmax(np.maximum.accumulate(port_equity_price.values) - port_equity_price.values) 
    # Code to pull start of drawdown period
    j = np.argmax(port_equity_price.values[:i]) 
    # I have done absolute value of drawdown to make it positive from negative
    port_max_drawdown = abs(100.0*(port_equity_price[i]-port_equity_price[j])/port_equity_price[j])

    port_equity_value = port_equity_price.index.values
    start_dt = pd.to_datetime(str(port_equity_value[j]))
    drawdown_start = start_dt.strftime ("%Y-%m-%d") 

    end_dt = pd.to_datetime(str(port_equity_value[i]))
    drawdown_end = end_dt.strftime ("%Y-%m-%d") 

    port_drawdown_duration=np.busday_count(drawdown_start, drawdown_end)

    return drawdown_start, drawdown_end, port_drawdown_duration, port_max_drawdown, 

# CODE TO PRINT MAX DRAWDOWN FOR ALL STOCK FOR WHOLE PERIOD 
for v in Stock_List:   
    f = portfoliomaxdrawdown(Global_Portfolio[v])
    print("Market: ",v,", Start Date: ", f[0], ", End Date: ", f[1])
    print("Duration in days: ",f[2], ", Max Drawdown Percent: %3.2f" %(f[3]))
    print()


# 4. Compare the global equity portfolio with the single market portfolios 
# (Hint: which is more attractive on the absolute return and on risk-adjusted basis)

# WHICH IS MORE ATTRACTIVE ? GLOBAL OR SINGLE
    
print(best_market,"is best Market with Highest Mean Return : %2.6f" %(best_return))
print(best_annual_market,"is best Market with Highest Average Annualised Return : %2.6f" %(best_annual_return))
print(riskiest_market,"is the Riskiest Market with Highest Overall Risk : %2.6f" %(highest_risk))
print(annualised_riskiest_market,"is the Riskiest Market with Highest Average Annualised Risk : %2.6f" %(max_annul_avgrisk))
print(best_rr_market,"is best Market with Highest Average Annualised Return Risk Ratio : %2.6f" %(highest_rratio))
print(best_sharpe_market,"is best Market with Highest Sharpe Ratio : %2.6f" %(highest_sratio))

   
print(port_best_market,"is best Global Portfolio Market with Highest Mean Return : %2.6f" %(port_best_return))
print(port_best_annual_market,"is best Global Portfolio Market with Highest Average Annualised Return : %2.6f" %(port_best_annual_return))
print(port_riskiest_market,"is the Riskiest Global Portfolio Market with Highest Overall Risk : %2.6f" %(port_highest_risk))
print(port_annualised_riskiest_market,"is the Riskiest Global Portfolio Market with Highest Average Annualised Risk : %2.6f" %(port_max_annul_avgrisk))
print(port_best_rr_market,"is best Global Portfolio Market with Highest Average Annualised Return Risk Ratio : %2.6f" %(port_highest_rratio))
print(port_best_sharpe_market,"is best Global Portfolio Market with Highest Sharpe Ratio : %2.6f" %(port_highest_sratio))

##################################################################################
# THE END
##################################################################################


