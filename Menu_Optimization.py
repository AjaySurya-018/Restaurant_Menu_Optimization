import streamlit as st
import pandas as pd
import pulp
import numpy as np

def load_and_preprocess_data():
    # Since we can't directly access the URL, let's create a sample dataset
    data = pd.read_excel("R:/OR/data/maindata.xlsx")
    data = data.drop_duplicates(subset=['RestaurantID', 'MenuItem'])
    return data

def calculate_selling_price(df):
    # Create profitability mapping
    profit_weights = {'Low': 2, 'Medium': 3, 'High': 4}
    
    # Calculate selling price
    df['SellingPrice'] = df.apply(lambda x: x['Price'] * profit_weights[x['Profitability']], axis=1)
    return df

def optimize_menu(df, restaurant_id, max_budget):
    # Fixed minimum items per category
    min_items_per_category = 1
    
    # Filter data for selected restaurant
    restaurant_df = df[df['RestaurantID'] == restaurant_id].copy()
    
    # Create the optimization problem
    prob = pulp.LpProblem(f"Menu_Optimization_{restaurant_id}", pulp.LpMaximize)
    
    # Create binary variables for each menu item
    # Use MenuItem as the unique identifier instead of index
    menu_vars = pulp.LpVariable.dicts("item",
                                    (row.MenuItem for _, row in restaurant_df.iterrows()),
                                    cat='Binary')
    
    # Objective function: Maximize total selling price
    prob += pulp.lpSum([menu_vars[row.MenuItem] * row.SellingPrice 
                       for _, row in restaurant_df.iterrows()])
    
    # Constraint 1: Budget constraint
    prob += pulp.lpSum([menu_vars[row.MenuItem] * row.Price 
                       for _, row in restaurant_df.iterrows()]) <= max_budget
    
    # Constraint 2: Minimum items per category (fixed at 1)
    for category in restaurant_df['MenuCategory'].unique():
        prob += pulp.lpSum([menu_vars[row.MenuItem] 
                           for _, row in restaurant_df.iterrows() 
                           if row.MenuCategory == category]) >= min_items_per_category
    
    # Solve the problem
    prob.solve()
    
    # Get selected items
    selected_items = []
    for _, row in restaurant_df.iterrows():
        if pulp.value(menu_vars[row.MenuItem]) == 1:
            selected_items.append({
                'MenuItem': row.MenuItem,
                'Category': row.MenuCategory,
                'Price': row.Price,
                'Profitability': row.Profitability,
                'SellingPrice': row.SellingPrice
            })
    
    return selected_items, pulp.value(prob.objective)

def display_restaurant_stats(df, restaurant_id):
    restaurant_df = df[df['RestaurantID'] == restaurant_id]
    
    st.subheader("Restaurant Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Menu Items", len(restaurant_df))
    with col2:
        st.metric("Average Price", f"${restaurant_df['Price'].mean():.2f}")
    with col3:
        st.metric("Menu Categories", len(restaurant_df['MenuCategory'].unique()))
    

def main():
    st.title("Restaurant-Specific Menu Optimization")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    df = calculate_selling_price(df)
    
    # Sidebar - Restaurant Selection
    st.sidebar.header("Restaurant Selection")
    restaurant_id = st.sidebar.selectbox(
        "Select Restaurant",
        options=sorted(df['RestaurantID'].unique()),
        format_func=lambda x: f"Restaurant {x}"
    )
    
    # Sidebar - Budget Parameter
    st.sidebar.header("Budget Parameter")
    max_budget = st.sidebar.slider("Maximum Budget ($)", 
                                 min_value=50, 
                                 max_value=300, 
                                 value=200, 
                                 step=10)
    
    # Display restaurant statistics
    display_restaurant_stats(df, restaurant_id)
    
    # Display restaurant's menu items
    st.subheader(f"Current Menu Items for {restaurant_id}")
    restaurant_df = df[df['RestaurantID'] == restaurant_id]
    st.dataframe(restaurant_df)
    
    # Run optimization
    if st.button("Optimize Menu"):
        selected_items, total_profit = optimize_menu(df, restaurant_id, max_budget)
        
        # Display results
        st.subheader("Optimized Menu Selection")
        if selected_items:
            results_df = pd.DataFrame(selected_items)
            st.dataframe(results_df)
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Items", len(selected_items))
            with col2:
                total_cost = sum(item['Price'] for item in selected_items)
                st.metric("Total Cost", f"${total_cost:.2f}")
            with col3:
                st.metric("Expected Profit", f"${total_profit:.2f}")
            
            # Display optimized category breakdown
            st.subheader("Optimized Category Distribution")
            category_counts = results_df['Category'].value_counts()
            st.bar_chart(category_counts)
            
            # Display profitability breakdown
            st.subheader("Profitability Distribution")
            profit_counts = results_df['Profitability'].value_counts()
            st.bar_chart(profit_counts)
        else:
            st.error("No feasible solution found with the given constraints!")

if __name__ == "__main__":
    main()