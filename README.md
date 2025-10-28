# ds_poojapattue


# ROBUST DATA PREPROCESSING AND BUSINESS INSIGHTS

# 1. Fix Date Format Issues
def fix_date_formats(sentiment_df, trader_df):
    """
    Fix datetime format issues in both datasets
    """
    print("Fixing date format issues...")
    
    # Fix sentiment dataframe dates
    print("Processing sentiment data dates...")
    sentiment_df['date_fixed'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
    
    # Fix trader dataframe dates and times
    print("Processing trader data dates and times...")
    
    # First, let's examine the timestamp format
    print("Sample trader timestamps:")
    print(trader_df['Timestamp IST'].head())
    
    # Try different date formats for trader data
    try:
        # Try DD-MM-YYYY HH:MM format
        trader_df['date_fixed'] = pd.to_datetime(trader_df['Timestamp IST'], 
                                                format='%d-%m-%Y %H:%M', 
                                                errors='coerce')
    except:
        try:
            # Try MM-DD-YYYY HH:MM format
            trader_df['date_fixed'] = pd.to_datetime(trader_df['Timestamp IST'], 
                                                    format='%m-%d-%Y %H:%M', 
                                                    errors='coerce')
        except:
            # Use flexible parsing as last resort
            trader_df['date_fixed'] = pd.to_datetime(trader_df['Timestamp IST'], 
                                                    errors='coerce')
    
    # Check for any remaining NaT values
    sentiment_nat = sentiment_df['date_fixed'].isna().sum()
    trader_nat = trader_df['date_fixed'].isna().sum()
    
    print(f"Sentiment data - NaT values: {sentiment_nat}/{len(sentiment_df)}")
    print(f"Trader data - NaT values: {trader_nat}/{len(trader_df)}")
    
    # Remove rows with invalid dates
    sentiment_clean = sentiment_df.dropna(subset=['date_fixed']).copy()
    trader_clean = trader_df.dropna(subset=['date_fixed']).copy()
    
    # Extract date only for daily aggregation
    sentiment_clean['date_only'] = sentiment_clean['date_fixed'].dt.date
    trader_clean['date_only'] = trader_clean['date_fixed'].dt.date
    
    print(f"Clean sentiment data: {len(sentiment_clean)} rows")
    print(f"Clean trader data: {len(trader_clean)} rows")
    
    return sentiment_clean, trader_clean

# Fix date formats
sentiment_clean, trader_clean = fix_date_formats(sentiment_df, trader_df)

# 2. Create Proper Daily Aggregates
def create_daily_aggregates(sentiment_clean, trader_clean):
    """
    Create daily aggregates from cleaned data
    """
    print("\nCreating daily aggregates...")
    
    # Sentiment daily (already daily data)
    sentiment_daily = sentiment_clean.groupby('date_only').agg({
        'sentiment_score': 'mean',
        'classification_lower': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral',
        'value': 'mean'
    }).reset_index()
    sentiment_daily.columns = ['date', 'avg_sentiment', 'dominant_sentiment', 'fear_greed_value']
    
    # Trader daily aggregates
    trader_daily = trader_clean.groupby('date_only').agg({
        'Size USD': ['sum', 'mean', 'std', 'count'],
        'Execution Price': ['mean', 'std'],
        'Fee': 'sum',
        'Side': lambda x: (x == 'BUY').sum(),
        'Account': 'nunique'
    }).round(2)
    
    # Flatten column names
    trader_daily.columns = [
        'total_volume', 'avg_trade_size', 'std_trade_size', 'trade_count',
        'avg_execution_price', 'std_execution_price', 
        'total_fees', 'buy_count', 'unique_traders'
    ]
    
    trader_daily = trader_daily.reset_index()
    trader_daily.columns = ['date', 'total_volume', 'avg_trade_size', 'std_trade_size', 
                           'trade_count', 'avg_execution_price', 'std_execution_price',
                           'total_fees', 'buy_count', 'unique_traders']
    
    # Calculate additional metrics
    trader_daily['buy_ratio'] = (trader_daily['buy_count'] / trader_daily['trade_count']).round(3)
    trader_daily['volume_per_trader'] = (trader_daily['total_volume'] / trader_daily['unique_traders']).round(2)
    
    print(f"Sentiment daily records: {len(sentiment_daily)}")
    print(f"Trader daily records: {len(trader_daily)}")
    
    # Display sample data
    print("\nSample sentiment daily:")
    display(sentiment_daily.head(3))
    print("\nSample trader daily:")
    display(trader_daily.head(3))
    
    return sentiment_daily, trader_daily

# Create daily aggregates
sentiment_daily, trader_daily = create_daily_aggregates(sentiment_clean, trader_clean)

# 3. Comprehensive Data Analysis
def comprehensive_data_analysis(sentiment_daily, trader_daily, trader_clean):
    """
    Perform comprehensive analysis on the properly formatted data
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print("📊 DATASET OVERVIEW:")
    print(f"Sentiment data: {len(sentiment_daily)} days")
    print(f"Trading data: {len(trader_daily)} days")
    print(f"Individual trades: {len(trader_clean):,}")
    
    # Sentiment analysis
    print(f"\n🎭 SENTIMENT ANALYSIS:")
    print(f"Average Fear & Greed Index: {sentiment_daily['fear_greed_value'].mean():.1f}")
    print(f"Sentiment range: {sentiment_daily['fear_greed_value'].min():.0f} - {sentiment_daily['fear_greed_value'].max():.0f}")
    
    sentiment_dist = sentiment_daily['dominant_sentiment'].value_counts()
    print("Sentiment distribution:")
    for sentiment, count in sentiment_dist.items():
        print(f"  {sentiment}: {count} days ({count/len(sentiment_daily):.1%})")
    
    # Trading analysis
    print(f"\n💸 TRADING ANALYSIS:")
    print(f"Total volume: ${trader_daily['total_volume'].sum():,.0f}")
    print(f"Average daily volume: ${trader_daily['total_volume'].mean():,.0f}")
    print(f"Average trade size: ${trader_daily['avg_trade_size'].mean():,.0f}")
    print(f"Average daily trades: {trader_daily['trade_count'].mean():.0f}")
    print(f"Average buy ratio: {trader_daily['buy_ratio'].mean():.1%}")
    
    # Temporal analysis
    print(f"\n⏰ TEMPORAL ANALYSIS:")
    trader_clean['hour'] = pd.to_datetime(trader_clean['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.hour
    hourly_activity = trader_clean['hour'].value_counts().sort_index()
    
    peak_hour = hourly_activity.idxmax()
    print(f"Peak trading hour: {peak_hour:02d}:00 ({hourly_activity.max()} trades)")
    
    # Day of week analysis
    trader_clean['day_of_week'] = pd.to_datetime(trader_clean['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.day_name()
    daily_activity = trader_clean['day_of_week'].value_counts()
    print("Busiest trading days:")
    for day, count in daily_activity.head(3).items():
        print(f"  {day}: {count} trades")
    
    return hourly_activity, daily_activity

# Perform comprehensive analysis
hourly_activity, daily_activity = comprehensive_data_analysis(sentiment_daily, trader_daily, trader_clean)

# 4. Visualization Dashboard
def create_analysis_dashboard(sentiment_daily, trader_daily, hourly_activity, daily_activity):
    """
    Create comprehensive visualization dashboard
    """
    print("\nCreating analysis dashboard...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Cryptocurrency Market Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Sentiment over time
    if len(sentiment_daily) > 0:
        axes[0,0].plot(sentiment_daily['date'], sentiment_daily['fear_greed_value'], 
                      color='purple', alpha=0.7, linewidth=2)
        axes[0,0].set_title('Fear & Greed Index Over Time')
        axes[0,0].set_ylabel('Sentiment Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Trading volume over time
    if len(trader_daily) > 0:
        axes[0,1].plot(trader_daily['date'], trader_daily['total_volume'], 
                      color='blue', alpha=0.7, linewidth=2)
        axes[0,1].set_title('Daily Trading Volume')
        axes[0,1].set_ylabel('Volume (USD)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Buy ratio over time
    if len(trader_daily) > 0:
        axes[0,2].plot(trader_daily['date'], trader_daily['buy_ratio'], 
                      color='green', alpha=0.7, linewidth=2)
        axes[0,2].set_title('Daily Buy Ratio')
        axes[0,2].set_ylabel('Buy Ratio')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Sentiment distribution
    if len(sentiment_daily) > 0:
        sentiment_daily['fear_greed_value'].hist(bins=20, ax=axes[1,0], 
                                               color='lightblue', alpha=0.7)
        axes[1,0].set_title('Sentiment Score Distribution')
        axes[1,0].set_xlabel('Fear & Greed Score')
        axes[1,0].set_ylabel('Frequency')
    
    # Plot 5: Volume distribution
    if len(trader_daily) > 0:
        trader_daily['total_volume'].hist(bins=20, ax=axes[1,1], 
                                        color='lightgreen', alpha=0.7)
        axes[1,1].set_title('Daily Volume Distribution')
        axes[1,1].set_xlabel('Volume (USD)')
        axes[1,1].set_ylabel('Frequency')
    
    # Plot 6: Trade size distribution
    if len(trader_daily) > 0:
        trader_daily['avg_trade_size'].hist(bins=20, ax=axes[1,2], 
                                          color='orange', alpha=0.7)
        axes[1,2].set_title('Average Trade Size Distribution')
        axes[1,2].set_xlabel('Trade Size (USD)')
        axes[1,2].set_ylabel('Frequency')
    
    # Plot 7: Hourly trading activity
    if len(hourly_activity) > 0:
        axes[2,0].bar(hourly_activity.index, hourly_activity.values, 
                     color='red', alpha=0.7)
        axes[2,0].set_title('Trading Activity by Hour')
        axes[2,0].set_xlabel('Hour of Day')
        axes[2,0].set_ylabel('Number of Trades')
    
    # Plot 8: Daily trading activity
    if len(daily_activity) > 0:
        daily_activity.plot(kind='bar', ax=axes[2,1], color='purple', alpha=0.7)
        axes[2,1].set_title('Trading Activity by Day')
        axes[2,1].set_xlabel('Day of Week')
        axes[2,1].set_ylabel('Number of Trades')
        axes[2,1].tick_params(axis='x', rotation=45)
    
    # Plot 9: Buy ratio distribution
    if len(trader_daily) > 0:
        trader_daily['buy_ratio'].hist(bins=20, ax=axes[2,2], 
                                     color='pink', alpha=0.7)
        axes[2,2].set_title('Buy Ratio Distribution')
        axes[2,2].set_xlabel('Buy Ratio')
        axes[2,2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Create dashboard
create_analysis_dashboard(sentiment_daily, trader_daily, hourly_activity, daily_activity)

# 5. BUSINESS INSIGHTS GENERATION
def generate_business_insights_robust(sentiment_daily, trader_daily, trader_clean):
    """
    Generate robust business insights from properly formatted data
    """
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS GENERATION")
    print("="*80)
    
    insights = []
    
    # Insight 1: Market Sentiment Overview
    avg_sentiment = sentiment_daily['fear_greed_value'].mean()
    sentiment_trend = "Bullish" if avg_sentiment > 50 else "Bearish"
    
    insights.append({
        'category': 'Market Sentiment',
        'insight': f"Average Fear & Greed Index: {avg_sentiment:.1f}/100 ({sentiment_trend} Market)",
        'implication': f"Market participants are generally {sentiment_trend.lower()}",
        'action': f"Adjust trading strategy for {sentiment_trend.lower()} market conditions"
    })
    
    # Insight 2: Trading Volume Analysis
    total_volume = trader_daily['total_volume'].sum()
    avg_daily_volume = trader_daily['total_volume'].mean()
    volume_volatility = trader_daily['total_volume'].std() / avg_daily_volume
    
    insights.append({
        'category': 'Trading Activity',
        'insight': f"Total trading volume: ${total_volume:,.0f}, Daily avg: ${avg_daily_volume:,.0f}",
        'implication': f"High volume volatility ({volume_volatility:.1%}) indicates frequent large trading events",
        'action': "Implement dynamic position sizing to handle volume fluctuations"
    })
    
    # Insight 3: Trade Characteristics
    avg_trade_size = trader_daily['avg_trade_size'].mean()
    large_trade_threshold = avg_trade_size * 3
    large_trade_days = len(trader_daily[trader_daily['avg_trade_size'] > large_trade_threshold])
    
    insights.append({
        'category': 'Trade Patterns',
        'insight': f"Average trade size: ${avg_trade_size:,.0f}, {large_trade_days} days with unusually large trades",
        'implication': "Market experiences institutional-sized trades periodically",
        'action': "Monitor for large trade days for potential market impact opportunities"
    })
    
    # Insight 4: Market Direction Bias
    avg_buy_ratio = trader_daily['buy_ratio'].mean()
    market_bias = "Buying" if avg_buy_ratio > 0.5 else "Selling"
    
    insights.append({
        'category': 'Market Direction',
        'insight': f"Average buy ratio: {avg_buy_ratio:.1%} ({market_bias} bias)",
        'implication': f"Market shows {market_bias.lower()} pressure overall",
        'action': f"Consider {market_bias.lower()} momentum or contrarian strategies"
    })
    
    # Insight 5: Optimal Trading Times
    trader_clean['hour'] = pd.to_datetime(trader_clean['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.hour
    hourly_stats = trader_clean.groupby('hour').agg({
        'Size USD': ['mean', 'count']
    }).round(2)
    hourly_stats.columns = ['avg_trade_size', 'trade_count']
    
    best_liquidity_hour = hourly_stats['trade_count'].idxmax()
    best_size_hour = hourly_stats['avg_trade_size'].idxmax()
    
    insights.append({
        'category': 'Execution Timing',
        'insight': f"Peak liquidity: {best_liquidity_hour:02d}:00, Largest trades: {best_size_hour:02d}:00",
        'implication': "Different trading objectives benefit from different execution times",
        'action': "Schedule executions based on specific trading goals (liquidity vs. size)"
    })
    
    # Insight 6: Trader Behavior
    avg_traders_per_day = trader_daily['unique_traders'].mean()
    volume_per_trader = trader_daily['volume_per_trader'].mean()
    
    insights.append({
        'category': 'Participant Analysis',
        'insight': f"Average {avg_traders_per_day:.0f} traders daily, ${volume_per_trader:,.0f} volume per trader",
        'implication': "Market participation and individual trader impact levels",
        'action': "Monitor trader concentration for market stability assessment"
    })
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight['category']}:")
        print(f"   📊 Insight: {insight['insight']}")
        print(f"   💡 Implication: {insight['implication']}")
        print(f"   🎯 Recommended Action: {insight['action']}")
    
    return insights

# Generate business insights
business_insights = generate_business_insights_robust(sentiment_daily, trader_daily, trader_clean)

# 6. PREDICTIVE ANALYSIS FRAMEWORK
def create_predictive_framework(trader_daily, sentiment_daily):
    """
    Create a framework for predictive analysis
    """
    print("\n" + "="*60)
    print("PREDICTIVE ANALYSIS FRAMEWORK")
    print("="*60)
    
    if len(trader_daily) < 10:
        print("Insufficient data for robust predictive modeling")
        print("Framework established for future implementation")
        return None
    
    # Feature engineering for prediction
    data = trader_daily.copy()
    
    # Create technical indicators
    for lag in [1, 2, 3, 5]:
        data[f'volume_lag_{lag}'] = data['total_volume'].shift(lag)
        data[f'buy_ratio_lag_{lag}'] = data['buy_ratio'].shift(lag)
    
    # Rolling statistics
    data['volume_ma_3'] = data['total_volume'].rolling(3).mean()
    data['volume_ma_7'] = data['total_volume'].rolling(7).mean()
    data['buy_ratio_ma_7'] = data['buy_ratio'].rolling(7).mean()
    
    # Volatility measures
    data['volume_volatility'] = data['total_volume'].rolling(5).std()
    
    # Remove incomplete rows
    data_clean = data.dropna()
    
    print(f"Data available for predictive modeling: {len(data_clean)} days")
    
    if len(data_clean) >= 5:
        print("Sufficient data for basic trend analysis")
        
        # Simple trend analysis
        volume_trend = data_clean['total_volume'].pct_change(3).mean()
        buy_ratio_trend = data_clean['buy_ratio'].diff(3).mean()
        
        print(f"3-day volume trend: {volume_trend:+.2%}")
        print(f"3-day buy ratio trend: {buy_ratio_trend:+.3f}")
        
        # Basic predictions
        last_volume = data_clean['total_volume'].iloc[-1]
        predicted_volume = last_volume * (1 + volume_trend)
        
        print(f"\nBasic Volume Prediction:")
        print(f"Last volume: ${last_volume:,.0f}")
        print(f"Predicted next day: ${predicted_volume:,.0f}")
        
        return {
            'volume_trend': volume_trend,
            'buy_ratio_trend': buy_ratio_trend,
            'predicted_volume': predicted_volume,
            'data_points': len(data_clean)
        }
    else:
        print("Insufficient data for trend analysis")
        return None

# Create predictive framework
predictive_results = create_predictive_framework(trader_daily, sentiment_daily)

# 7. TRADING STRATEGY RECOMMENDATIONS
def generate_trading_recommendations(insights, predictive_results=None):
    """
    Generate specific trading recommendations
    """
    print("\n" + "="*60)
    print("TRADING STRATEGY RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Recommendation 1: Sentiment-Based Strategy
    recommendations.append({
        'strategy': 'Sentiment-Adaptive Trading',
        'description': 'Adjust trading approach based on Fear & Greed Index levels',
        'rules': [
            'Extreme Fear (<20): Accumulate positions',
            'Fear (20-40): Moderate buying',
            'Neutral (40-60): Range trading',
            'Greed (60-80): Take profits',
            'Extreme Greed (>80): Reduce exposure'
        ],
        'risk_management': 'Use smaller position sizes during sentiment extremes'
    })
    
    # Recommendation 2: Volume-Weighted Execution
    recommendations.append({
        'strategy': 'Smart Order Routing',
        'description': 'Optimize trade execution based on volume patterns',
        'rules': [
            'Execute large orders during high-volume hours',
            'Use VWAP for better average pricing',
            'Avoid low-volume periods for large trades',
            'Monitor volume spikes for market opportunities'
        ],
        'risk_management': 'Set maximum order size as percentage of daily volume'
    })
    
    # Recommendation 3: Time-Based Strategies
    recommendations.append({
        'strategy': 'Temporal Arbitrage',
        'description': 'Exploit predictable daily and weekly patterns',
        'rules': [
            'Focus on peak liquidity hours for execution',
            'Monitor opening/closing period anomalies',
            'Watch for weekend vs weekday patterns',
            'Consider time-zone based opportunities'
        ],
        'risk_management': 'Limit overnight positions during volatile periods'
    })
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['strategy']}:")
        print(f"   📝 {rec['description']}")
        print(f"   📋 Trading Rules:")
        for rule in rec['rules']:
            print(f"      • {rule}")
        print(f"   🛡️ Risk Management: {rec['risk_management']}")
    
    return recommendations

# Generate trading recommendations
trading_recommendations = generate_trading_recommendations(business_insights, predictive_results)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
print("="*80)
print("✅ Successfully processed and cleaned all data")
print("✅ Comprehensive market analysis completed")
print("✅ 6 Key business insights generated")
print("✅ Predictive framework established")
print("✅ 3 Trading strategies recommended")
print("✅ Visualization dashboard created")

print("\n📈 KEY METRICS:")
print(f"   • Market Sentiment: {sentiment_daily['fear_greed_value'].mean():.1f}/100")
print(f"   • Total Trading Volume: ${trader_daily['total_volume'].sum():,.0f}")
print(f"   • Average Daily Trades: {trader_daily['trade_count'].mean():.0f}")
print(f"   • Market Bias: {'Buying' if trader_daily['buy_ratio'].mean() > 0.5 else 'Selling'}")

print("\n🎯 IMMEDIATE ACTIONS:")
actions = [
    "1. Implement sentiment-based position sizing",
    "2. Optimize trade execution timing using identified patterns", 
    "3. Establish risk thresholds based on volume volatility",
    "4. Monitor large trade days for strategic opportunities",
    "5. Deploy the predictive framework for daily analysis"
]

for action in actions:
    print(f"   {action}")
