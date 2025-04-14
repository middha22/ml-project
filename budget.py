import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import os

# Global variables
df = pd.DataFrame(columns=['description', 'amount', 'category', 'date'])
pred_model = None
categories = ["Food/Grocery", "Housing/Rent", "Utilities", "Transportation", 
              "Shopping", "Entertainment", "Healthcare", "Others"]
FILE_PATH = "expenses.csv"  # File to save and load data

# Step 1: Load and Preprocess Data
def preprocess_data(df):
    # Encode categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Feature extraction from description
    tfidf = TfidfVectorizer(max_features=5)  # Limit to 5 features for small dataset
    
    description_features = tfidf.fit_transform(df['description']).toarray()
    description_df = pd.DataFrame(description_features, columns=tfidf.get_feature_names_out())
    
    # Combine features
    df = pd.concat([df, description_df], axis=1)
    return df, label_encoder

# Step 2: Train the Expense Categorization Model
def train_categorization_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 3: Train the Expense Prediction Model
def train_prediction_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to save data to file
def save_data():
    global df
    try:
        df.to_csv(FILE_PATH, index=False)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {str(e)}")

# Function to load data from file
def load_data():
    global df
    if os.path.exists(FILE_PATH):
        try:
            df = pd.read_csv(FILE_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

# Function to add an expense
def add_expense():
    global df
    
    # Create a new top-level window
    add_window = tk.Toplevel()
    add_window.title("Add New Expense")
    add_window.geometry("400x300")
    
    # Description field
    tk.Label(add_window, text="Description:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    description_entry = tk.Entry(add_window, width=30)
    description_entry.grid(row=0, column=1, padx=10, pady=5)
    
    # Amount field
    tk.Label(add_window, text="Amount:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    amount_entry = tk.Entry(add_window, width=30)
    amount_entry.grid(row=1, column=1, padx=10, pady=5)
    
    # Category dropdown
    tk.Label(add_window, text="Category:").grid(row=2, column=0, padx=10, pady=5, sticky='e')
    category_var = tk.StringVar(value=categories[0])
    category_dropdown = ttk.Combobox(add_window, textvariable=category_var, 
                                     values=categories, state="readonly")
    category_dropdown.grid(row=2, column=1, padx=10, pady=5)
    
    # Date field
    tk.Label(add_window, text="Date (YYYY-MM-DD):").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    date_entry = tk.Entry(add_window, width=30)
    date_entry.grid(row=3, column=1, padx=10, pady=5)
    
    # Submit button
    def submit_expense():
        description = description_entry.get().strip()
        amount = amount_entry.get().strip()
        category = category_var.get()
        date = date_entry.get().strip()
        
        # Validate inputs
        if not description:
            messagebox.showerror("Error", "Description cannot be empty!")
            return
            
        try:
            amount = float(amount)
            if amount <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Amount must be a positive number!")
            return
            
        if not is_valid_date(date):
            messagebox.showerror("Error", "Date must be in the format YYYY-MM-DD!")
            return
            
        # Add to dataframe
        new_expense = pd.DataFrame({
            'description': [description],
            'amount': [amount],
            'category': [category],
            'date': [date]
        })
        
        global df
        df = pd.concat([df, new_expense], ignore_index=True)
        
        # Save the updated data
        save_data()
        
        # Update the display
        display_expenses()
        add_window.destroy()
        messagebox.showinfo("Success", "Expense added successfully!")
    
    submit_button = tk.Button(add_window, text="Submit Expense", command=submit_expense)
    submit_button.grid(row=4, column=1, padx=10, pady=10, sticky='e')

# Function to display all expenses
def display_expenses():
    global df
    
    # Clear previous display
    for widget in expense_display_frame.winfo_children():
        widget.destroy()
    
    if df.empty:
        tk.Label(expense_display_frame, text="No expenses added yet.").pack()
        return
    
    # Create a scrollable frame
    canvas = tk.Canvas(expense_display_frame)
    scrollbar = ttk.Scrollbar(expense_display_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Add column headers
    headers = ["Date", "Description", "Category", "Amount"]
    for i, header in enumerate(headers):
        tk.Label(scrollable_frame, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=5, pady=2)
    
    # Add expense data
    for idx, row in df.iterrows():
        tk.Label(scrollable_frame, text=row['date']).grid(row=idx+1, column=0, padx=5, pady=2)
        tk.Label(scrollable_frame, text=row['description']).grid(row=idx+1, column=1, padx=5, pady=2)
        tk.Label(scrollable_frame, text=row['category']).grid(row=idx+1, column=2, padx=5, pady=2)
        tk.Label(scrollable_frame, text=f"${row['amount']:.2f}").grid(row=idx+1, column=3, padx=5, pady=2)

# Function to predict next month's expenses
def predict_expenses():
    global df, pred_model
    if df.empty:
        messagebox.showwarning("Warning", "No expenses added yet. Please add expenses first.")
        return
    
    # Check if there are at least 2 entries to train the model
    if len(df) < 2:
        messagebox.showwarning("Warning", "At least 2 expenses are required to predict next month's expenses.")
        return
    
    # Preprocess data
    df['month'] = pd.to_datetime(df['date']).dt.month
    X_pred = df[['amount', 'month']]
    y_pred = df['amount'].shift(-1).dropna()  # Predict next month's expense
    X_pred = X_pred.iloc[:-1, :]
    
    # Train the prediction model
    pred_model = train_prediction_model(X_pred, y_pred)
    
    # Predict next month's expense
    last_month = df['month'].max()
    new_expense = np.array([[df['amount'].mean(), last_month + 1]])  # Use mean amount for prediction
    predicted_expense = pred_model.predict(new_expense)
    
    # Update the prediction label
    prediction_label.config(text=f"Predicted Expense for Next Month: ${predicted_expense[0]:.2f}")

# Function to display summary
def display_summary():
    global df
    if df.empty:
        messagebox.showwarning("Warning", "No expenses added yet. Please add expenses first.")
        return
    
    # Total Expenses
    total_expenses = df['amount'].sum()
    
    # Average Monthly Expense
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_expenses = df.groupby('month')['amount'].sum()
    avg_monthly_expense = monthly_expenses.mean()
    
    # Expense Breakdown by Category
    category_expenses = df.groupby('category')['amount'].sum()
    summary = (f"Total Expenses: ${total_expenses:.2f}\n"
               f"Average Monthly Expense: ${avg_monthly_expense:.2f}\n\n"
               "Expense Breakdown by Category:\n")
    for category, amount in category_expenses.items():
        summary += f"{category}: ${amount:.2f}\n"
    
    # Update the summary label
    summary_label.config(text=summary)

# Function to validate date format (YYYY-MM-DD)
def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d')
        return True
    except ValueError:
        return False

# Main GUI Application
class BudgetTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Budget Tracker")
        self.root.geometry("600x600")
        
        # Load existing data
        load_data()
        
        # Create main frames
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(fill=tk.X)
        
        global expense_display_frame
        expense_display_frame = ttk.Frame(root, padding="10")
        expense_display_frame.pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.Frame(root, padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add Buttons
        self.add_button = ttk.Button(control_frame, text="Add Expense", command=add_expense, width=20)
        self.add_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.predict_button = ttk.Button(control_frame, text="Predict Next Month", command=predict_expenses, width=20)
        self.predict_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.summary_button = ttk.Button(control_frame, text="Display Summary", command=display_summary, width=20)
        self.summary_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.exit_button = ttk.Button(control_frame, text="Exit", command=root.quit, width=20)
        self.exit_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Info display labels
        global prediction_label, summary_label
        prediction_label = ttk.Label(info_frame, text="Prediction will appear here", wraplength=550)
        prediction_label.pack(anchor='w', pady=5)
        
        summary_label = ttk.Label(info_frame, text="Summary will appear here", wraplength=550)
        summary_label.pack(anchor='w', pady=5)
        
        # Initial expense display
        display_expenses()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = BudgetTrackerApp(root)
    root.mainloop()
